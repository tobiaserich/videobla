#!/usr/bin/env python3
"""
Standalone FastAPI Server für LongCat-Video
Läuft OHNE Runpod - direkt auf deinem Server!

Usage:
    python standalone_server.py --port 8000
    
    curl -X POST http://localhost:8000/generate \
      -H "Content-Type: application/json" \
      -d '{"prompt": "A cat riding a skateboard", "duration": 5}'
"""

import os
import sys
import argparse
from pathlib import Path

# LongCat-Video zum Path hinzufügen
REPO_ROOT = Path(__file__).parent
LONGCAT_PATH = REPO_ROOT / "LongCat-Video"
if LONGCAT_PATH.exists():
    sys.path.insert(0, str(LONGCAT_PATH))

# Handler-Funktionen importieren
sys.path.insert(0, str(REPO_ROOT / "serverless"))
from handler import generate_video, upload_to_storage, load_model

# FastAPI
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("❌ FastAPI not installed!")
    print("   Install: pip install fastapi uvicorn python-multipart")
    sys.exit(1)


app = FastAPI(
    title="LongCat-Video API",
    description="Text-to-Video Generation Server",
    version="1.0.0"
)


class GenerateRequest(BaseModel):
    prompt: str
    task: str = "text_to_video"
    duration: int = 5
    resolution: str = "720p"
    fps: int = 30
    guidance_scale: float = 7.5
    num_inference_steps: int = 50


@app.on_event("startup")
async def startup_event():
    """Model beim Server-Start laden (Warm Start)"""
    print("\n" + "🚀" * 30)
    print("Starting LongCat-Video Standalone Server")
    print("🚀" * 30 + "\n")
    
    load_model()
    print("\n✅ Server ready!\n")


@app.get("/")
async def root():
    """API Info"""
    return {
        "name": "LongCat-Video API",
        "version": "1.0.0",
        "endpoints": {
            "POST /generate": "Generate video from prompt",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health():
    """Health Check"""
    import torch
    
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }


@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Video generieren
    
    Returns base64-encoded video für kleine Files (<10MB)
    oder File-Path für größere Videos
    """
    
    try:
        # Video generieren
        result = generate_video(
            prompt=request.prompt,
            task=request.task,
            duration=request.duration,
            resolution=request.resolution,
            fps=request.fps,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps
        )
        
        # Upload/Encode
        video_result = upload_to_storage(result["video_path"])
        
        response = {
            "success": True,
            "video": video_result,
            "metadata": {
                "prompt": request.prompt,
                "resolution": result["resolution"],
                "width": result["width"],
                "height": result["height"],
                "duration": result["duration"],
                "fps": result["fps"],
                "frames": result["frames"],
                "mode": result["mode"]
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@app.get("/download/{filename}")
async def download_video(filename: str):
    """Video-Download Endpoint (für große Files)"""
    import tempfile
    from pathlib import Path
    
    video_path = Path(tempfile.gettempdir()) / "longcat_output" / filename
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=filename
    )


def main():
    parser = argparse.ArgumentParser(description="LongCat-Video Standalone Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"\n🌐 Starting server on http://{args.host}:{args.port}")
    print(f"📖 API Docs: http://{args.host}:{args.port}/docs\n")
    
    uvicorn.run(
        "standalone_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
