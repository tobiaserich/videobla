#!/usr/bin/env python3
"""
LongCat-Video Runpod Serverless Client

Usage:
    python client.py --prompt "A cat riding a skateboard" --duration 5
"""

import os
import sys
import time
import json
import argparse
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")


class RunpodClient:
    """Client für Runpod Serverless API"""
    
    def __init__(self, api_key: Optional[str] = None, endpoint_id: Optional[str] = None):
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        self.endpoint_id = endpoint_id or os.getenv("RUNPOD_ENDPOINT_ID")
        
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY not found. Set it in .env or pass as argument")
        if not self.endpoint_id:
            raise ValueError("RUNPOD_ENDPOINT_ID not found. Set it in .env or pass as argument")
        
        self.base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_video(
        self,
        prompt: str,
        task: str = "text_to_video",
        duration: int = 5,
        resolution: str = "720p",
        fps: int = 30,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Video generieren via Runpod Serverless
        
        Args:
            prompt: Text prompt für Video-Generierung
            task: "text_to_video", "image_to_video", "video_continuation"
            duration: Video-Länge in Sekunden
            resolution: "720p", "1080p"
            fps: Frames per second (30 empfohlen)
            image_path: Pfad zum Bild (für image_to_video)
            video_path: Pfad zum Video (für video_continuation)
        
        Returns:
            Dict mit Video-URL und Metadaten
        """
        
        payload = {
            "input": {
                "task": task,
                "prompt": prompt,
                "duration": duration,
                "resolution": resolution,
                "fps": fps,
                **kwargs
            }
        }
        
        # Bild/Video als base64 oder URL übergeben (je nach Implementation)
        if image_path and task == "image_to_video":
            payload["input"]["image_path"] = image_path
        if video_path and task == "video_continuation":
            payload["input"]["video_path"] = video_path
        
        print(f"🚀 Starte Video-Generierung auf Runpod...")
        print(f"   Task: {task}")
        print(f"   Prompt: {prompt}")
        print(f"   Duration: {duration}s @ {fps}fps")
        
        # Job starten
        response = requests.post(
            f"{self.base_url}/run",
            headers=self.headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        job_data = response.json()
        
        job_id = job_data.get("id")
        if not job_id:
            raise RuntimeError(f"Failed to start job: {job_data}")
        
        print(f"✅ Job gestartet: {job_id}")
        
        # Auf Completion warten
        return self._wait_for_completion(job_id)
    
    def _wait_for_completion(self, job_id: str, timeout: int = 600) -> Dict[str, Any]:
        """Warte auf Job-Completion"""
        
        start_time = time.time()
        status_url = f"{self.base_url}/status/{job_id}"
        
        print("⏳ Warte auf Completion...")
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job {job_id} exceeded timeout of {timeout}s")
            
            response = requests.get(status_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            status_data = response.json()
            
            status = status_data.get("status")
            
            if status == "COMPLETED":
                output = status_data.get("output", {})
                print(f"✨ Video generiert!")
                print(f"   Video URL: {output.get('video_url', 'N/A')}")
                print(f"   Execution Time: {output.get('execution_time', 'N/A')}s")
                return output
            
            elif status == "FAILED":
                error = status_data.get("error", "Unknown error")
                raise RuntimeError(f"Job failed: {error}")
            
            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                elapsed = int(time.time() - start_time)
                print(f"   Status: {status} (elapsed: {elapsed}s)", end="\r")
                time.sleep(2)
            
            else:
                print(f"   Unknown status: {status}")
                time.sleep(2)
    
    def download_video(self, video_url: str, output_path: str = "output.mp4"):
        """Video von URL herunterladen"""
        
        print(f"⬇️  Downloading video to {output_path}...")
        
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Video saved to: {output_path.absolute()}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="LongCat-Video Runpod Client")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--task", type=str, default="text_to_video", 
                       choices=["text_to_video", "image_to_video", "video_continuation"],
                       help="Generation task type")
    parser.add_argument("--duration", type=int, default=5, help="Video duration in seconds")
    parser.add_argument("--resolution", type=str, default="720p", choices=["720p", "1080p"])
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--image", type=str, help="Input image path (for image_to_video)")
    parser.add_argument("--video", type=str, help="Input video path (for video_continuation)")
    
    args = parser.parse_args()
    
    try:
        # Client erstellen
        client = RunpodClient()
        
        # Video generieren
        result = client.generate_video(
            prompt=args.prompt,
            task=args.task,
            duration=args.duration,
            resolution=args.resolution,
            fps=args.fps,
            image_path=args.image,
            video_path=args.video
        )
        
        # Video herunterladen
        if "video_url" in result:
            client.download_video(result["video_url"], args.output)
        else:
            print("⚠️  No video URL in result")
            print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
