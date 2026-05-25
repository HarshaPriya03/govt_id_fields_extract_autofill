import os
import re
import json
import base64
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# ─── Set your Gemini API key here ───────────────────────────────────────────
GEMINI_API_KEY = "your api key"

genai.configure(api_key=GEMINI_API_KEY)

generation_config = {
    "temperature": 0.1,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 2048,
}

model = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config=generation_config
)
# ────────────────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image(file) -> Image.Image:
    """Load and lightly normalize image for Gemini."""
    img = Image.open(file).convert("RGB")
    # Resize if too large to save tokens (max 1600px on longest side)
    max_side = 1600
    w, h = img.size
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    return img

def clean_json(text: str) -> dict:
    """Strip markdown fences and parse JSON safely."""
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(text)
    except Exception:
        # Try to find JSON object in the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {}

# ─── Prompts ────────────────────────────────────────────────────────────────

AADHAAR_PROMPT = """
You are an expert Aadhaar OCR extractor.

Image may contain:
- front side
- back side
- both sides together
- rotated image
- blurry image
- partial image
- Telugu + English text

Return ONLY JSON.

Extract:
- name
- dob
- gender
- aadhaar_number
- address
- pincode

Always prefer English values.

Never return garbage text.

If field missing return null.

{
  "name": null,
  "dob": null,
  "gender": null,
  "aadhaar_number": null,
  "address": null,
  "pincode": null,
  "confidence": {
    "name": 0,
    "dob": 0,
    "gender": 0,
    "aadhaar_number": 0,
    "address": 0,
    "pincode": 0
  }
}
"""

PAN_PROMPT = """
You are an expert Indian KYC document parser. Analyze this PAN card image.

RULES:
- The image may be rotated, blurry, or partially cropped — still extract what is visible.
- PAN card has TWO names: the card holder's own name AND the father's name. DO NOT confuse them.
- The card holder's name appears FIRST (usually the larger/bolder text after "Income Tax Department").
- Father's name appears SECOND, explicitly labeled "Father's Name" or preceded by "S/O" or "Son of" or "D/O".
- NEVER swap these two names. If unsure which is which, return null for father_name.
- PAN number: exactly 10 characters, format AAAAA0000A (5 letters, 4 digits, 1 letter). All uppercase.
- DOB format: DD/MM/YYYY.
- Names should be in English only, no regional language text.
- Return ONLY raw JSON, no explanation, no markdown, no extra text.

Return this exact JSON structure:
{
  "name": "card holder's own name or null",
  "father_name": "father's name only (NOT the card holder's name) or null",
  "dob": "DD/MM/YYYY or null",
  "pan_number": "10-char PAN or null",
  "confidence": {
    "name": 0-100,
    "father_name": 0-100,
    "dob": 0-100,
    "pan_number": 0-100
  }
}
"""

# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/extract", methods=["POST"])
def extract():
    doc_type = request.form.get("doc_type")  # aadhaar_front | aadhaar_back | pan
    if not doc_type:
        return jsonify({"error": "doc_type is required"}), 400

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use JPG, PNG, WEBP, or BMP"}), 400

    try:
        img = load_image(file)
    except Exception as e:
        return jsonify({"error": f"Could not read image: {str(e)}"}), 400

    # Pick prompt
    prompt_map = {
        "aadhaar": AADHAAR_PROMPT,
        "pan": PAN_PROMPT,
    }
    prompt = prompt_map.get(doc_type)
    if not prompt:
        return jsonify({"error": "Invalid doc_type"}), 400

    try:
        response = model.generate_content([prompt, img])
        raw = response.text.strip()
        data = clean_json(raw)

        if not data:
            return jsonify({"error": "Could not parse response from Gemini", "raw": raw}), 500

        # Post-process: strip leading/trailing whitespace from all string values
        def clean_val(v):
            if isinstance(v, str):
                v = v.strip()
                return v if v.lower() not in ("null", "none", "", "n/a", "not visible") else None
            return v

        cleaned = {}
        for k, v in data.items():
            if k == "confidence":
                cleaned[k] = v
            else:
                cleaned[k] = clean_val(v)

        return jsonify({"success": True, "doc_type": doc_type, "data": cleaned})

    except Exception as e:
        return jsonify({"error": f"Gemini API error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
