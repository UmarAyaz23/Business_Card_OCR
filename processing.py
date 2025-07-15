import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from PIL import Image
import cv2
from pyzbar.pyzbar import decode
import base64
import easyocr
from paddleocr import PaddleOCR

import os
import re
import json
from dotenv import load_dotenv

import groq
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

#--------------------------------------------------------------------------API INITIALIZATION--------------------------------------------------------------------------
load_dotenv()
LANCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = groq.Groq(api_key = GROQ_API_KEY)


#--------------------------------------------------------------------------IMAGE PROCESSING--------------------------------------------------------------------------
def extract_img_data_paddleOCR(img_files, min_avg_confidence = 0.7):
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    all_rows = []

    for img_path in img_files:
        result = ocr.ocr(img_path, cls=True)

        word_entries = []
        confidences = []

        qr_result = decode_qr_code_cv2(img_path)

        for line in result:
            for word_info in line:
                text = word_info[1][0]
                conf = round(word_info[1][1], 3)
                confidences.append(conf)

                word_entries.append({
                    'Image': img_path,
                    'Text': text,
                    'Confidence': conf,
                    'QR_Code': qr_result
                })

        avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

        # Skip image if average confidence is too low
        if avg_conf < min_avg_confidence:
            print(f"⚠️ Skipped {img_path} due to low average confidence ({avg_conf})")
            continue

        # Add avg score to each word row
        for entry in word_entries:
            entry['Average Confidence'] = avg_conf
            all_rows.append(entry)

    return pd.DataFrame(all_rows)


def decode_qr_code_pyzbar(img_path):
    try:
        image = Image.open(img_path)
        qr_codes = decode(image)

        if qr_codes:
            qr_data = qr_codes[0].data.decode('utf-8').strip()
            return qr_data if qr_data else "QR Code Present"
        else:
            return "Unavailable"
    except Exception as e:
        print(f"⚠️ QR decode error for {img_path}: {e}")
        return "QR Code Present"


def decode_qr_code_cv2(img_path):
    try:
        image = cv2.imread(img_path)

        # Preprocessing: grayscale + sharpening
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)  # smooths noise while keeping edges

        detector = cv2.QRCodeDetector()

        # Use multi detection (in case there are multiple QR codes)
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(gray)

        if retval:
            decoded_texts = [text.strip() for text in decoded_info if text.strip()]
            if decoded_texts:
                return " | ".join(decoded_texts)
            else:
                return "QR Code Present"
        else:
            return "Unavailable"
    except Exception as e:
        print(f"⚠️ Error decoding QR code from {img_path}: {e}")
        return "QR Code Present"


def structure_data_LLM(text):
    llm = ChatGroq(
        model = "llama-3.1-8b-instant",
        temperature = 0.2,
    )

    messages = [
        {
            'role': 'system', 'content': (
                "You are an AI specialized in identifying what type of data is being given to you"
                "The invoice may contain text in English or Arabic, or both."
                "The input text may be noisy or OCR-generated.\n"
                "There may be multiple values per field — do not exclude any. If there are multiple contact numbers, include them in the same column with a comma that separates them\n"
                "If the website name looks something like: 'WWW[name]', then make it like 'www.[name]'.\n"
                "Your response must always be a valid JSON object, formatted exactly as follows:"
                "\n```json\n"
                "{"
                "\n  \"Company Name\": \"<string>\","
                "\n  \"Full Name\": \"<string>\","
                "\n  \"Designation\": \"<string>\","
                "\n  \"Contact Number\": \"<string>\","
                "\n  \"Fax Number\": \"<string>\","
                "\n  \"Email\": \"<string>\","
                "\n  \"Website\": \"<string>\","
                "\n  \"Address\": <string>,"
                "\n  \"Social Media Handle Name\": <string>,"
                "\n```"
                "\nEnsure the JSON structure remains consistent and does not wrap data in extra keys."
            )
        },
        {
            "role": "user", "content": 
                f"Extract data from the following OCR text:\n{text}"
        }
    ]

    response_text = llm.invoke(messages).content.strip()
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        cleaned_json = match.group(1).strip()
    else:
        cleaned_json = response_text.strip()

    try:
        return json.loads(cleaned_json)
    except json.JSONDecodeError:
        print("❌ Failed to decode JSON:")
        print(cleaned_json)
        return {}


def get_value(data, keys, default="Unavailable"):
    for key in keys:
        if key in data:
            return data[key]
    return default


def process_all_images(output_file):
    upload_dir = 'uploads'
    img_files = glob(os.path.join(upload_dir, '*'))

    if not img_files:
        print("❌ No images found in uploads.")
        return pd.DataFrame()

    # Clear Excel file if exists
    if os.path.exists(output_file):
        os.remove(output_file)

    img_data = extract_img_data_paddleOCR(img_files, min_avg_confidence=0.5)

    if img_data.empty or 'Image' not in img_data.columns or 'Text' not in img_data.columns:
        print("❌ OCR failed or no readable text.")
        return pd.DataFrame()

    grouped = img_data.groupby('Image').agg({
        'Text': lambda x: ' '.join(x),
        'QR_Code': 'first'
    }).reset_index()

    structured_rows = []

    for _, row in grouped.iterrows():
        ocr_text = row['Text']
        image_path = row['Image']

        extracted = structure_data_LLM(ocr_text)

        structured_rows.append({
            'Image': image_path,
            'Company_Name': get_value(extracted, ['Company Name', 'Company_Name', 'CompanyName']),
            'Full_Name': get_value(extracted, ['Full Name', 'Full_Name']),
            'Designation': get_value(extracted, ['Designation']),
            'Contact_Number': get_value(extracted, ['Contact Number', 'Contact_Number', 'ContactNumber']),
            'Fax_Number': get_value(extracted, ['Fax Number', 'Fax_Number', 'FaxNumber']),
            'Email': get_value(extracted, ['Email']),
            'Website': get_value(extracted, ['Website']),
            'Address': get_value(extracted, ['Address']),
            'Social_Media_Handle_Name': get_value(extracted, ['Social Media Handle Name', 'Social_Media_Handle_Name', 'SocialMediaHandleName']),
            'QR_Code': row['QR_Code'],
        })

    new_df = pd.DataFrame(structured_rows).astype(str).fillna('Unavailable')
    new_df.to_excel(output_file, index=False)
    print(f"✅ Excel updated at: {output_file}")

    return new_df