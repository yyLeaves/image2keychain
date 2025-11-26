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

class ModelResponse(BaseModel):
    success: bool
    model_url: str
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

@app.post("/convert-to-3d", response_model=ModelResponse)
async def convert_to_3d(
    local_path: str = Form(...),
    depth_div_width: float = Form(..., description="Ratio of desired depth to width (e.g., 0.5)", gt=0),
    aspect_ratio: float = Form(1.0, description="Height to width ratio (default: 1.0)", gt=0)
):
    """
    Converts a depth map image to a 3D STL model.
    
    Parameters:
    - local_path: Path to the depth map image (usually from generate-silhouette or edit-silhouette)
    - depth_div_width: Ratio of depth to width (e.g., 0.5 for 10cm deep and 20cm wide)
    - aspect_ratio: Height to width ratio (1.0 = original proportions, >1.0 = taller, <1.0 = wider)
    """
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        # Convert the depth map to 3D model
        output_path = service.convert_depth_to_stl(
            local_path, 
            depth_div_width, 
            aspect_ratio
        )
        
        # Return URL to download the STL file
        url_path = f"/static/models/{os.path.basename(output_path)}"
        return {"success": True, "model_url": url_path, "local_path": output_path}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-model/{filename}")
async def download_model(filename: str):
    """Downloads the generated STL model file."""
    file_path = os.path.join("static", "models", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return FileResponse(
        path=file_path,
        media_type='application/vnd.ms-pki.stl',
        filename=filename
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)