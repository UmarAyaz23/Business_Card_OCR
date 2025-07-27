from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from processing import process_image_from_bytes

app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/ping")
async def ping():
    return {"status": "Backend is running"}

@app.post("/api/extract-card-data")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    
    if file.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(status_code=400, content={"error": "Only JPG and PNG images are supported."})
    
    try:
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image.")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Image decoding failed: {e}"})

    extracted_data = process_image_from_bytes(img)
    
    if not extracted_data:
        return JSONResponse(status_code=400, content={"error": "No data could be extracted from the image."})

    return {
        "filename": file.filename,
        "extracted": extracted_data
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)