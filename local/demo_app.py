#!/usr/bin/env python3
"""
Streamlit Demo App für LongCat-Video

Run: streamlit run demo_app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Client importieren
sys.path.append(str(Path(__file__).parent))
from client import RunpodClient

st.set_page_config(
    page_title="LongCat-Video Generator",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 LongCat-Video Generator")
st.markdown("**Serverless Video Generation** powered by Runpod")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

api_key = st.sidebar.text_input("Runpod API Key", type="password", 
                                help="Get it from runpod.io/console/settings")
endpoint_id = st.sidebar.text_input("Endpoint ID",
                                   help="Your serverless endpoint ID")

st.sidebar.markdown("---")

task = st.sidebar.selectbox(
    "Task",
    ["text_to_video", "image_to_video", "video_continuation"],
    help="Video generation mode"
)

duration = st.sidebar.slider("Duration (seconds)", 1, 30, 5)
resolution = st.sidebar.selectbox("Resolution", ["720p", "1080p"])
fps = st.sidebar.selectbox("FPS", [24, 30, 60], index=1)

# Main Interface
if task == "text_to_video":
    st.header("📝 Text-to-Video")
    prompt = st.text_area(
        "Enter your prompt",
        placeholder="A cat riding a skateboard through a neon city at night...",
        height=100
    )
    
elif task == "image_to_video":
    st.header("🖼️ Image-to-Video")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    prompt = st.text_input("Motion prompt", placeholder="The cat starts walking...")
    
elif task == "video_continuation":
    st.header("🎥 Video-Continuation")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    prompt = st.text_input("Continuation prompt", placeholder="The scene continues with...")

# Cost Estimation
st.sidebar.markdown("---")
st.sidebar.header("💰 Cost Estimate")

# Grobe Schätzung (30fps, ~1s pro Sekunde Generierung)
estimated_time = duration * 1.0  # Sekunden
estimated_cost = (estimated_time / 60) * 0.34  # $0.34/min für RTX 4090

st.sidebar.metric("Estimated Time", f"~{estimated_time:.0f}s")
st.sidebar.metric("Estimated Cost", f"${estimated_cost:.3f}")

# Generate Button
if st.button("🚀 Generate Video", type="primary", use_container_width=True):
    
    if not api_key or not endpoint_id:
        st.error("❌ Please provide API Key and Endpoint ID in the sidebar")
    elif task == "text_to_video" and not prompt:
        st.error("❌ Please enter a prompt")
    else:
        try:
            with st.spinner("Generating video... This may take a while ⏳"):
                
                # Client erstellen
                client = RunpodClient(api_key=api_key, endpoint_id=endpoint_id)
                
                # Video generieren
                result = client.generate_video(
                    prompt=prompt,
                    task=task,
                    duration=duration,
                    resolution=resolution,
                    fps=fps
                )
                
                st.success("✅ Video generated successfully!")
                
                # Results anzeigen
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Execution Time", f"{result.get('execution_time', 'N/A')}s")
                with col2:
                    actual_cost = (result.get('execution_time', 0) / 60) * 0.34
                    st.metric("Actual Cost", f"${actual_cost:.3f}")
                
                # Video URL
                video_url = result.get("video_url")
                if video_url:
                    st.markdown(f"**Video URL:** {video_url}")
                    
                    # Download Button
                    if st.button("⬇️ Download Video"):
                        output_path = f"output_{task}.mp4"
                        client.download_video(video_url, output_path)
                        st.success(f"Downloaded to {output_path}")
                
                # Metadata
                with st.expander("📊 Metadata"):
                    st.json(result)
                    
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
**💡 Tips:**
- Longer videos cost more (pay-per-second)
- Keep workers warm for faster generation
- Use 720p for testing, 1080p for production
- Batch multiple videos to save on cold starts
""")

st.markdown("Built with ❤️ using [Runpod](https://runpod.io) and [LongCat-Video](https://huggingface.co/meituan-longcat/LongCat-Video)")
