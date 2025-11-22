import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from services import ImageService


app = FastAPI(title="Keychain Generator API")
service = ImageService()

# Mount static directory to serve generated images
app.mount("/static", StaticFiles(directory="static"), name="static")

class AnalysisResponse(BaseModel):
    success: bool
    ratio_string: Optional[str] = None
    data: Optional[dict] = None
    error: Optional[str] = None

class ImageResponse(BaseModel):
    success: bool
    image_url: str
    local_path: str

@app.post("/upload", response_model=ImageResponse)
async def upload_image(file: UploadFile = File(...)):
    """Uploads an image and converts it to PNG."""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    try:
        contents = await file.read()
        saved_path = service.convert_to_png(contents, file.filename)
        
        # Return a URL that can be accessed via the browser
        url_path = f"/static/uploads/{os.path.basename(saved_path)}"
        return {"success": True, "image_url": url_path, "local_path": saved_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(local_path: str = Form(...)):
    """Analyzes the uploaded image for proportions."""
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="Image file not found")
        
    try:
        result = service.analyze_proportions(local_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-silhouette", response_model=ImageResponse)
async def generate_silhouette(local_path: str = Form(...)):
    """Generates a black and white silhouette from the image."""
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="Image file not found")
        
    try:
        output_path = service.generate_silhouette(local_path)
        url_path = f"/static/processed/{os.path.basename(output_path)}"
        return {"success": True, "image_url": url_path, "local_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/edit-silhouette", response_model=ImageResponse)
async def edit_silhouette(
    local_path: str = Form(...), 
    instructions: str = Form("")
):
    """
    Uploads an image with red marks and applies edits based on instructions.
    The file uploaded here should be the silhouette modified by the user.
    """
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    try:
        output_path = service.edit_silhouette(local_path, instructions=instructions)
        
        url_path = f"/static/processed/{os.path.basename(output_path)}"
        return {"success": True, "image_url": url_path, "local_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)