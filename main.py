from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager

from processing import process_all_images 
import shutil
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🔁 STARTUP CLEANUP (when app starts)
    upload_dir = "uploads"
    if os.path.exists(upload_dir):
        for file in os.listdir(upload_dir):
            try:
                os.remove(os.path.join(upload_dir, file))
            except Exception as e:
                print(f"⚠️ Could not delete {file}: {e}")

    excel_file = "structured_output.xlsx"
    if os.path.exists(excel_file):
        try:
            os.remove(excel_file)
        except Exception as e:
            print(f"⚠️ Could not delete Excel file: {e}")

    print("🚀 App startup: uploads and excel file cleaned.")
    yield  # ⬅ App runs here
    # 🔚 SHUTDOWN CLEANUP (when app stops)
    if os.path.exists(upload_dir):
        for file in os.listdir(upload_dir):
            try:
                os.remove(os.path.join(upload_dir, file))
            except Exception as e:
                print(f"⚠️ Could not delete {file}: {e}")
    if os.path.exists(excel_file):
        try:
            os.remove(excel_file)
        except Exception as e:
            print(f"⚠️ Could not delete Excel file: {e}")
    print("🧹 App shutdown: uploads and excel file cleaned.")

app = FastAPI(lifespan=lifespan)

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Ensure uploads dir exists
os.makedirs("uploads", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/results", response_class=HTMLResponse)
async def show_results(request: Request):
    output_file = "structured_output.xlsx"
    df = process_all_images(output_file)

    return templates.TemplateResponse("results.html", {
        "request": request,
        "data": df.to_dict(orient="records")  # Converts dataframe to list of dicts
    })

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return RedirectResponse(url="/results", status_code=303)