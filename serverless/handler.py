"""
Runpod Serverless Handler für LongCat-Video - MIT ECHTER INTEGRATION!

Dieser Handler lädt das echte LongCat-Video Model und generiert Videos.
"""

import os
import sys
import json
import base64
import time
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# Deaktiviere hf_transfer - kann bei großen Downloads Probleme machen
os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER', None)

# Flash-attn Fallback installieren falls nicht verfügbar
try:
    import flash_attn
except ImportError:
    try:
        from flash_attn_fallback import install_fallback
        install_fallback()
    except ImportError:
        print("⚠️  flash-attn fallback not available")

import torch
import runpod


# Model wird beim Container-Start geladen (warm start)
MODEL = None
DEVICE = None


def load_model():
    """Lädt LongCat-Video Model beim Container-Start"""
    global MODEL, DEVICE
    
    if MODEL is not None:
        print("Model already loaded")
        return MODEL
    
    print("=" * 60)
    print("Loading LongCat-Video model...")
    print("=" * 60)
    
    # GPU Detection
    if torch.cuda.is_available():
        DEVICE = "cuda"
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        DEVICE = "cpu"
        print("⚠️  No GPU found, using CPU (VERY slow!)")
    
    # LongCat-Video laden
    try:
        # LongCat-Video Repository zum Python Path hinzufügen
        longcat_repo_path = os.path.join(os.path.dirname(__file__), "..", "LongCat-Video")
        if not os.path.exists(longcat_repo_path):
            longcat_repo_path = "/app/LongCat-Video"  # Docker path
        
        if os.path.exists(longcat_repo_path):
            sys.path.insert(0, longcat_repo_path)
            print(f"Added {longcat_repo_path} to Python path")
        
        # LongCat-Video importieren - korrekter Import-Pfad!
        try:
            from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
            from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
            from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
            from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
            from transformers import AutoTokenizer, UMT5EncoderModel
            has_longcat = True
            print("✅ LongCat-Video modules imported successfully")
        except ImportError as e:
            print(f"⚠️  LongCat-Video module not found: {e}")
            print("   Falling back to DUMMY mode")
            has_longcat = False
        
        if has_longcat:
            # Model-ID auf HuggingFace
            model_id = "meituan-longcat/LongCat-Video"
            # Bevorzuge /workspace (hat mehr Platz auf Runpod)
            cache_dir = os.getenv("HF_HOME", "/workspace/.cache/huggingface" if os.path.exists("/workspace") else os.path.expanduser("~/.cache/huggingface"))
            
            print(f"Loading model: {model_id}")
            print(f"Cache directory: {cache_dir}")
            
            try:
                # Komponenten einzeln laden (wie im Demo-Script)
                print("Loading tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, 
                    subfolder="tokenizer",
                    cache_dir=cache_dir
                )
                
                print("Loading text encoder...")
                text_encoder = UMT5EncoderModel.from_pretrained(
                    model_id,
                    subfolder="text_encoder",
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    cache_dir=cache_dir
                )
                
                print("Loading VAE...")
                vae = AutoencoderKLWan.from_pretrained(
                    model_id,
                    subfolder="vae",
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    cache_dir=cache_dir
                )
                
                print("Loading scheduler...")
                scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                    model_id,
                    subfolder="scheduler",
                    cache_dir=cache_dir
                )
                
                print("Loading DiT transformer...")
                dit = LongCatVideoTransformer3DModel.from_pretrained(
                    model_id,
                    subfolder="dit",
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    cache_dir=cache_dir
                )
                
                # Defensive: recursively set cp_split_hw for all modules that need it
                def set_cp_split_hw_recursive(module, value=(1, 1)):
                    """Recursively set cp_split_hw to avoid NoneType errors in rope_3d and other modules."""
                    if hasattr(module, 'cp_split_hw'):
                        if module.cp_split_hw is None:
                            module.cp_split_hw = value
                    for child in module.children():
                        set_cp_split_hw_recursive(child, value)
                
                print("Setting cp_split_hw defaults...")
                set_cp_split_hw_recursive(dit)
                
                # Pipeline zusammenbauen
                print("Assembling pipeline...")
                MODEL = LongCatVideoPipeline(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    vae=vae,
                    scheduler=scheduler,
                    dit=dit,
                )
                MODEL.to(DEVICE)
                
                # Optional: Compile für schnellere Inference
                if os.getenv("ENABLE_COMPILE", "false").lower() == "true":
                    print("Compiling model...")
                    MODEL = torch.compile(MODEL)
                
                print(f"✅ LongCat-Video loaded successfully!")
                print(f"   Device: {DEVICE}")
                print(f"   Model type: {type(MODEL)}")
                print(f"   Cache: {cache_dir}")
                
                return MODEL
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                traceback.print_exc()
                print("   Falling back to DUMMY mode")
                has_longcat = False
        
        # Fallback: Dummy Model
        print("📦 Using DUMMY mode for testing")
        MODEL = {
            "type": "dummy",
            "device": DEVICE,
            "message": "LongCat-Video not installed or model not downloaded"
        }
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        traceback.print_exc()
        # Dummy Model als Fallback
        MODEL = {
            "type": "dummy",
            "device": DEVICE,
            "error": str(e)
        }
    
    print("=" * 60)
    return MODEL


def generate_video_real(model, prompt, task, duration, width, height, fps, **kwargs):
    """Generiert Video mit echtem LongCat-Video Model"""
    
    total_frames = duration * fps
    
    print(f"🎬 Generating with real LongCat-Video model...")
    print(f"   Frames: {total_frames}, Size: {width}x{height}, FPS: {fps}")
    
    # LongCat-Video nutzt spezielle generate-Methoden
    if task == "text_to_video":
        output = model.generate_t2v(
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt", ""),
            height=height,
            width=width,
            num_frames=total_frames,
            num_inference_steps=int(kwargs.get("num_inference_steps", 50)),
            guidance_scale=float(kwargs.get("guidance_scale", 4.0)),
        )
    
    elif task == "image_to_video":
        image_path = kwargs.get("image_path")
        if not image_path:
            raise ValueError("image_path required for image_to_video")
        
        from PIL import Image
        image = Image.open(image_path)
        
        output = model.generate_i2v(
            prompt=prompt,
            image=image,
            negative_prompt=kwargs.get("negative_prompt", ""),
            height=height,
            width=width,
            num_frames=total_frames,
            num_inference_steps=int(kwargs.get("num_inference_steps", 50)),
            guidance_scale=float(kwargs.get("guidance_scale", 4.0)),
        )
    
    elif task == "video_continuation":
        video_path = kwargs.get("video_path")
        if not video_path:
            raise ValueError("video_path required for video_continuation")
        
        # TODO: Implementiere video continuation
        # Braucht spezielle Video-Loading-Logik
        raise NotImplementedError("Video continuation not yet implemented")
    
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Output ist eine Liste von frames
    return output[0] if isinstance(output, list) else output


def generate_video_dummy(prompt, task, duration, width, height, fps):
    """Generiert Dummy Video für Testing ohne echtes Model"""
    
    import numpy as np
    import cv2
    
    print(f"🎨 Generating DUMMY video (model not loaded)...")
    
    output_dir = Path(tempfile.gettempdir()) / "longcat_output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"dummy_{task}_{int(time.time())}.mp4"
    
    total_frames = duration * fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Text overlay
        cv2.putText(frame, f"{prompt[:35]}", (50, height//2 - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Task: {task}", (50, height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, f"Frame {i+1}/{total_frames}", (50, height//2 + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.putText(frame, "[DUMMY MODE - Model not loaded]", (50, height//2 + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        cv2.putText(frame, "Install LongCat-Video for real generation", (50, height//2 + 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
        out.write(frame)
    
    out.release()
    return output_path


def generate_video(
    prompt: str,
    task: str = "text_to_video",
    duration: int = 5,
    resolution: str = "720p",
    fps: int = 30,
    **kwargs
) -> Dict[str, Any]:
    """
    Generiert Video mit LongCat-Video (oder Dummy wenn Model nicht geladen)
    """
    
    print("=" * 60)
    print(f"🎥 Video Generation Request")
    print(f"   Task: {task}")
    print(f"   Prompt: {prompt[:60]}...")
    print(f"   Duration: {duration}s @ {fps}fps")
    print(f"   Resolution: {resolution}")
    print("=" * 60)
    
    model = load_model()
    
    # Resolution mapping
    res_map = {
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "480p": (854, 480),
    }
    width, height = res_map.get(resolution, (1280, 720))
    
    # Output directory
    output_dir = Path(tempfile.gettempdir()) / "longcat_output"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Echtes Model oder Dummy?
        is_dummy = isinstance(model, dict) and model.get("type") == "dummy"
        
        if not is_dummy:
            # ECHTE LongCat-Video Generation
            output_frames = generate_video_real(
                model, prompt, task, duration, width, height, fps, **kwargs
            )
            
            # Video speichern (output_frames ist numpy array)
            import numpy as np
            from torchvision.io import write_video
            
            output_path = output_dir / f"video_{task}_{int(time.time())}.mp4"
            
            # Konvertiere zu Tensor und speichere
            output_tensor = torch.from_numpy(np.array(output_frames))
            output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
            write_video(str(output_path), output_tensor, fps=fps, video_codec="libx264", options={"crf": "18"})
            
            print(f"✅ Real video generated: {output_path}")
            
        else:
            # DUMMY Generation
            output_path = generate_video_dummy(
                prompt, task, duration, width, height, fps
            )
            
            print(f"⚠️  Dummy video generated: {output_path}")
        
        result = {
            "video_path": str(output_path),
            "resolution": resolution,
            "width": width,
            "height": height,
            "duration": duration,
            "fps": fps,
            "frames": duration * fps,
            "task": task,
            "mode": "dummy" if is_dummy else "real",
        }
        
        print("=" * 60)
        print("✨ Generation completed successfully!")
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"❌ Video generation failed: {e}")
        traceback.print_exc()
        raise


def upload_to_storage(video_path: str) -> Dict[str, str]:
    """
    Upload Video zu Storage oder encode als base64
    
    Für Testing: Encode als base64 und zurückgeben
    Für Production: Upload zu S3/R2 und return URL
    """
    
    # Check file size
    file_size = os.path.getsize(video_path)
    print(f"Video size: {file_size / 1024 / 1024:.2f} MB")
    
    # Option 1: Base64 encode (gut für kleine Videos, <10MB)
    if file_size < 10 * 1024 * 1024:  # < 10MB
        print("Encoding video as base64...")
        with open(video_path, "rb") as f:
            video_bytes = f.read()
            video_b64 = base64.b64encode(video_bytes).decode('utf-8')
        
        return {
            "type": "base64",
            "data": video_b64,
            "filename": os.path.basename(video_path)
        }
    
    # Option 2: File path (für große Videos)
    # TODO: Implement S3/R2 upload für Production
    else:
        print("⚠️  Video too large for base64, returning file path")
        return {
            "type": "file",
            "path": video_path,
            "message": "Video too large - implement S3 upload for production"
        }


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runpod Serverless Handler
    
    Event format:
    {
        "input": {
            "task": "text_to_video",
            "prompt": "A cat riding a skateboard",
            "duration": 5,
            "resolution": "720p",
            "fps": 30,
            "guidance_scale": 7.5,  # optional
            "num_inference_steps": 50  # optional
        }
    }
    """
    
    start_time = time.time()
    
    try:
        print("\n" + "=" * 60)
        print("🚀 Runpod Handler invoked")
        print("=" * 60)
        
        # Input validieren
        input_data = event.get("input", {})
        
        if not input_data:
            raise ValueError("No input data provided")
        
        task = input_data.get("task", "text_to_video")
        prompt = input_data.get("prompt")
        
        if not prompt:
            raise ValueError("Prompt is required")
        
        # Video generieren
        result = generate_video(
            prompt=prompt,
            task=task,
            duration=input_data.get("duration", 5),
            resolution=input_data.get("resolution", "720p"),
            fps=input_data.get("fps", 30),
            guidance_scale=input_data.get("guidance_scale", 7.5),
            num_inference_steps=input_data.get("num_inference_steps", 50),
        )
        
        # Video uploaden
        video_result = upload_to_storage(result["video_path"])
        
        execution_time = time.time() - start_time
        
        output = {
            "video": video_result,
            "metadata": {
                "task": task,
                "prompt": prompt,
                "resolution": result["resolution"],
                "width": result["width"],
                "height": result["height"],
                "duration": result["duration"],
                "fps": result["fps"],
                "frames": result["frames"],
                "mode": result.get("mode", "unknown"),
            },
            "execution_time": round(execution_time, 2),
            "success": True
        }
        
        print(f"\n✅ Handler completed in {execution_time:.2f}s")
        print("=" * 60 + "\n")
        
        return output
        
    except Exception as e:
        print(f"\n❌ Handler error: {e}")
        traceback.print_exc()
        
        execution_time = time.time() - start_time
        
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "execution_time": round(execution_time, 2)
        }


# Runpod Serverless Entry Point
if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("Starting Runpod Serverless Handler for LongCat-Video")
    print("🚀" * 30 + "\n")
    
    # Model vorladen (Warm Start)
    load_model()
    
    # Runpod Handler starten
    print("\n✅ Handler ready, waiting for requests...\n")
    runpod.serverless.start({"handler": handler})
