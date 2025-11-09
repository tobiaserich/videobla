# LongCat-Video Server - Quickstart Guide

## 🚀 One-Command Setup

```bash
git pull && ./setup.sh && ./start_server.sh
```

That's it! 🎉

---

## 📋 Detailed Steps

### 1. Clone & Setup
```bash
git clone https://github.com/tobiaserich/videobla.git
cd videobla
chmod +x setup.sh start_server.sh
./setup.sh
```

**What it does:**
- ✅ Detects system PyTorch (e.g., 2.8+ on Runpod) and uses `--system-site-packages`
- ✅ Creates `/workspace/.cache/huggingface` for large model storage (if in /workspace)
- ✅ Installs all dependencies including `flash-attn` (3-5 min compile time)
- ✅ Clones LongCat-Video repo and configures PYTHONPATH
- ✅ Sets up environment variables in `~/.bashrc` for persistence

### 2. Start Server
```bash
./start_server.sh           # Foreground
./start_server.sh -b        # Background (logs to /tmp/server.log)
```

**Server will:**
- Load LongCat-Video model on startup (warm start, ~2-3 minutes)
- Listen on `http://0.0.0.0:8000`
- Auto-configure HF cache and PYTHONPATH

### 3. Test Generation
```bash
# Quick test (4 frames, ~30 seconds)
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a cat walking in a garden",
    "height": 384,
    "width": 384,
    "num_frames": 4,
    "num_inference_steps": 10,
    "guidance_scale": 3.0
  }'

# Full quality (8 frames, ~2-3 minutes)
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a cat walking in a garden",
    "height": 512,
    "width": 512,
    "num_frames": 8,
    "num_inference_steps": 20,
    "guidance_scale": 3.0
  }'
```

---

## 🔧 Configuration

### Environment Variables (auto-configured by setup.sh)
```bash
# Fast downloads (requires hf-transfer package)
export HF_HUB_ENABLE_HF_TRANSFER=1

# Cache directory (avoids filling up root partition)
export HF_HOME=/workspace/.cache/huggingface

# LongCat-Video Python imports
export PYTHONPATH="${PYTHONPATH}:/path/to/LongCat-Video"
```

### Performance Tuning
**Faster generation (for testing):**
- `num_frames: 4` instead of 8
- `num_inference_steps: 10` instead of 20
- `height/width: 384` instead of 512

**Enable torch.compile (first run slower, subsequent runs faster):**
```bash
export ENABLE_COMPILE=true
./start_server.sh
```

---

## 📦 Key Features

✅ **Automatic setup detection:**
- Detects system PyTorch 2.8+ → uses `--system-site-packages`
- Detects `/workspace` → configures HF cache there (100GB+ free)
- Detects GPU → installs flash-attn for 2-3x faster attention

✅ **One-command updates:**
```bash
git pull && ./setup.sh  # Re-run setup after pull
```

✅ **Background mode:**
```bash
./start_server.sh -b
tail -f /tmp/server.log
```

✅ **Health check:**
```bash
curl http://localhost:8000/health
```

---

## 🐛 Troubleshooting

### Model download fails with "No space left"
- **Solution:** Setup script auto-detects `/workspace` and configures cache there
- **Manual:** `export HF_HOME=/workspace/.cache/huggingface`

### "flash-attn" compilation fails
- **Not critical!** Server will use slower attention fallback
- Flash-attn gives ~2-3x speedup but requires CUDA + compile time

### Import errors (longcat_video.*)
- **Solution:** Run `./setup.sh` again (sets PYTHONPATH in ~/.bashrc)
- **Manual:** `export PYTHONPATH="$PYTHONPATH:$(pwd)/LongCat-Video"`

### Server already running
```bash
pkill -f standalone_server.py
./start_server.sh
```

### Check logs
```bash
tail -f /tmp/server.log          # If started with -b
journalctl -u videobla -f        # If using systemd
```

---

## 📊 System Requirements

- **GPU:** NVIDIA with 24GB+ VRAM (tested on RTX PRO 6000 Blackwell 102GB)
- **Disk:** 30GB+ free space for model cache
- **RAM:** 16GB+ recommended
- **CUDA:** 12.1+ (PyTorch 2.5.1+cu121 or system torch 2.8+cu128)

---

## 🎯 API Endpoints

### `POST /generate`
Generate video from text prompt.

**Request:**
```json
{
  "prompt": "a cat walking",
  "height": 512,           // Must be divisible by 16
  "width": 512,            // Must be divisible by 16
  "num_frames": 8,         // 4-16 recommended
  "num_inference_steps": 20,  // 10-30 range
  "guidance_scale": 3.0    // 2.0-5.0 recommended
}
```

**Response:**
```json
{
  "video_base64": "...",   // Base64 encoded MP4
  "status": "success"
}
```

### `GET /health`
Check server status.

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA RTX PRO 6000..."
}
```

---

## 🔄 Update Workflow

```bash
# 1. Stop server
pkill -f standalone_server

# 2. Pull latest changes
git pull

# 3. Re-run setup (updates deps if needed)
./setup.sh

# 4. Restart server
./start_server.sh -b
```

---

## 💡 Tips

1. **First generation is slow** (~2-3 min model loading + 2-3 min generation)
   - Subsequent generations are faster (model stays loaded)

2. **Use small params for testing:**
   - `num_frames: 4, num_inference_steps: 10, height/width: 384`

3. **Background mode for production:**
   - `./start_server.sh -b` and monitor logs with `tail -f /tmp/server.log`

4. **Runpod proxy URL:**
   - Server auto-exposes on Runpod proxy (e.g., `https://xxx-8000.proxy.runpod.net/`)

---

**Maintained by:** [@tobiaserich](https://github.com/tobiaserich)  
**Model:** [LongCat-Video by Meituan](https://huggingface.co/meituan-longcat/LongCat-Video)
