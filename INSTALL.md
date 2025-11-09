# 🎬 LongCat-Video Server Setup

Komplette Anleitung für Installation und Deployment auf eigenem Server.

## 🚀 Quick Start

```bash
# 1. Repository klonen
git clone https://github.com/tobiaserich/videobla.git
cd videobla

# 2. Setup-Script ausführen (macht ALLES automatisch!)
chmod +x setup.sh
./setup.sh

# 3. Environment aktivieren
source venv/bin/activate

# 4. API Keys konfigurieren
nano .env  # Füge RUNPOD_API_KEY ein

# 5. Server starten
python standalone_server.py
```

Das wars! Server läuft auf `http://localhost:8000` 🎉

---

## 📋 Was macht das Setup-Script?

1. **System-Check**: Python, Git, GPU detection
2. **Virtual Environment**: Erstellt isolierte Python-Umgebung
3. **PyTorch**: Installiert GPU- oder CPU-Version automatisch
4. **Dependencies**: Alle nötigen Packages
5. **LongCat-Video**: Klon repo und installiere Model
6. **Configuration**: .env Template erstellen
7. **Test**: Prüft ob alles funktioniert

---

## 🖥️ Server-Modi

### Option A: Standalone Server (eigener Server mit GPU)

```bash
# Server starten
python standalone_server.py --port 8000

# Video generieren
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat riding a skateboard",
    "duration": 5,
    "resolution": "720p"
  }'

# API Docs anzeigen
open http://localhost:8000/docs
```

**Requirements:**

- NVIDIA GPU mit mind. 12GB VRAM
- CUDA 12.1+
- ~30GB Festplatte für Model

---

### Option B: Runpod Serverless (empfohlen für Production!)

```bash
# Client verwenden (nutzt Runpod GPUs)
python local/client.py \
  --prompt "A dog playing guitar" \
  --duration 5 \
  --output video.mp4

# Kosten: ~$0.15 pro 5s Video
# Keine eigene GPU nötig!
```

**Vorteile:**

- Pay-per-second billing
- Auto-scaling
- RTX 4090 / A100 GPUs
- Kein Server-Management

---

## 🔧 Installation ohne Script (Manuell)

Falls das Setup-Script nicht funktioniert:

```bash
# 1. Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 2. PyTorch (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Dependencies
cd serverless
pip install -r requirements.txt
cd ..

# 4. LongCat-Video
git clone https://github.com/meituan-longcat/LongCat-Video.git
cd LongCat-Video
pip install diffusers transformers accelerate pillow opencv-python
cd ..

# 5. FastAPI für Server
pip install fastapi uvicorn python-multipart

# 6. .env konfigurieren
cp .env.example .env
nano .env
```

---

## 🐛 Troubleshooting

### Problem: `flash-attn` build failed

```bash
# LongCat ohne flash-attn installieren
cd LongCat-Video
grep -v "flash-attn" requirements.txt > requirements_safe.txt
pip install -r requirements_safe.txt
```

### Problem: Torch version conflict

```bash
# PyTorch neu installieren
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Problem: GPU nicht erkannt

```bash
# CUDA check
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Falls false: CUDA Toolkit installieren
# https://developer.nvidia.com/cuda-downloads
```

### Problem: Server startet nicht

```bash
# Dependencies prüfen
pip list | grep -E "fastapi|uvicorn|torch"

# Fehlende installieren
pip install fastapi uvicorn
```

---

## 📊 System-Anforderungen

### Minimum (Dummy-Mode):

- Python 3.9+
- 4GB RAM
- Keine GPU (nutzt CPU)

### Empfohlen (echte Videos):

- Python 3.10+
- NVIDIA GPU mit 12GB+ VRAM (RTX 3090, RTX 4090, A100)
- 32GB RAM
- 50GB Festplatte
- CUDA 12.1+

### Alternative (Runpod Serverless):

- Beliebiger Computer
- Internetverbindung
- Runpod API Key

---

## 🌍 Production Deployment

### Docker (empfohlen):

Der GitHub Actions Workflow baut automatisch:

```bash
# Push zu GitHub → Auto-Deploy auf Runpod
git push origin main
```

### Systemd Service (eigener Server):

```bash
# Service erstellen
sudo nano /etc/systemd/system/longcat-video.service
```

```ini
[Unit]
Description=LongCat-Video API Server
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/videobla
Environment="PATH=/path/to/videobla/venv/bin"
ExecStart=/path/to/videobla/venv/bin/python standalone_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Service aktivieren
sudo systemctl enable longcat-video
sudo systemctl start longcat-video
sudo systemctl status longcat-video
```

---

## 📞 Support

- **Issues**: https://github.com/tobiaserich/videobla/issues
- **LongCat-Video**: https://huggingface.co/meituan-longcat/LongCat-Video
- **Runpod Docs**: https://docs.runpod.io/

---

## 📝 Lizenz

MIT License - Siehe LICENSE file
