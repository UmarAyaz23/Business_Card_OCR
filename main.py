from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from bson import json_util

import numpy as np
import cv2
from processing import process_image_from_bytes
from dotenv import load_dotenv
import os

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import gridfs


#--------------------------------------------------------------------------API INITIALIZATION--------------------------------------------------------------------------
load_dotenv()
MONGODB_USERNAME = os.getenv("MONGODB_USERNAME").strip()
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD").strip()
uri = f"mongodb+srv://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@business-card-ocr.yi8tqmh.mongodb.net/?retryWrites=true&w=majority&appName=Business-Card-OCR"
client = MongoClient(uri, server_api=ServerApi('1'))

db = client["business_card_ocr"]
fs = gridfs.GridFS(db)
results_collection = db["extracted_data"]


#--------------------------------------------------------------------------FASTAPI INITIALIZATION--------------------------------------------------------------------------
app = FastAPI()

# Add IPs in allow_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#--------------------------------------------------------------------------PING STATUS--------------------------------------------------------------------------
@app.get("/api/ping")
async def ping():
    return {"status": "Backend is running"}


#--------------------------------------------------------------------------UPLOAD API--------------------------------------------------------------------------
@app.post("/api/extract-card-data")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    if not file.content_type in ["image/jpeg", "image/png"]:
        return JSONResponse(status_code=400, content={"error": "Only JPG and PNG images are supported."})
    
    # Check and remove existing file with same filename
    existing_file = db.fs.files.find_one({"filename": file.filename})
    if existing_file:
        fs.delete(existing_file["_id"])
        results_collection.delete_many({"filename": file.filename})  # Optional: delete old extracted data

    # Save new image in GridFS
    file_id = fs.put(contents, filename=file.filename)

    # Convert image bytes to OpenCV image
    try:
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Image decoding failed: {e}"})

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})

    extracted_data = process_image_from_bytes(img)

    if not extracted_data:
        return JSONResponse(status_code=400, content={"error": "Could not extract any data from the image."})
    
    # Add metadata and store result in MongoDB
    safe_extracted = extracted_data.copy()
    safe_extracted["_image_id"] = str(file_id)
    safe_extracted["filename"] = file.filename
    inserted_id = results_collection.insert_one(safe_extracted).inserted_id

    return Response(
        content=json_util.dumps({
            "message": "Image processed and data stored.",
            "image_id": file_id,
            "data_id": inserted_id,
            "extracted": safe_extracted
        }),
        media_type="application/json"
    )