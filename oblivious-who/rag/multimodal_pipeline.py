import base64
import os
from PIL import Image
import pytesseract
import requests


TESSERACT_PATHS = (
    os.environ.get("TESSERACT_CMD", ""),
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)

for tesseract_path in TESSERACT_PATHS:
    if tesseract_path and os.path.isfile(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        break

OLLAMA_VISION_URL = "http://localhost:11434/api/chat"


def extract_text_tesseract(image_path: str) -> str:
    """Tier 1: Fast local Tesseract OCR for typed scans."""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img).strip()
        return text
    except Exception as e:
        print(f"⚠️ Tesseract error: {e}")
        return ""


def extract_vision_qwen(
    image_path: str, model_name: str = "qwen2-vl:7b"
) -> str:
    """Tier 2: Qwen2-VL vision model for handwritten logs & P&ID engineering schematics."""
    if not os.path.exists(image_path):
        return "Image file not found."

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": "Extract all readable text, equipment labels, measurements, and key findings from this image.",
                "images": [encoded_image],
            }
        ],
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_VISION_URL, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            return result.get("message", {}).get("content", "").strip()
        else:
            return f"Vision API error HTTP {response.status_code}"
    except Exception as e:
        return f"Failed to connect to local vision model: {e}"


def process_multimodal_document(
    image_path: str, min_char_threshold: int = 50
) -> dict:
    """Hybrid 2-Tier Pipeline Handler."""
    print(f"🔍 Processing document image: {image_path}")

    # Step 1: Try Tesseract OCR first
    ocr_text = extract_text_tesseract(image_path)

    if len(ocr_text) >= min_char_threshold:
        print("🟢 Tier 1 (Tesseract OCR) successful.")
        return {"method": "Tesseract_OCR", "extracted_text": ocr_text}

    # Step 2: Fallback to Qwen2-VL / Qwen3-VL Vision Model
    print("🟡 Tier 1 yield low. Falling back to Tier 2 (Qwen2-VL Vision)...")
    vlm_text = extract_vision_qwen(image_path)
    return {"method": "Qwen_Vision_VLM", "extracted_text": vlm_text}


if __name__ == "__main__":
    test_img = os.path.join("knowledge", "inspection", "sample_inspection.png")
    if os.path.exists(test_img):
        res = process_multimodal_document(test_img)
        print("\n--- Extracted Content ---")
        print(f"Method: {res['method']}")
        print(f"Content:\n{res['extracted_text']}")
    else:
        print("Place an image at 'knowledge/inspection/sample_inspection.png' to test.")