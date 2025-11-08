# 🚀 LongCat-Video Serverless - Deployment Guide

## 📋 Voraussetzungen

- Runpod Account (https://runpod.io)
- Docker installiert (für lokales Testing)
- Docker Hub Account (oder andere Container Registry)

## 🎯 Schritt-für-Schritt Anleitung

### 1️⃣ Runpod Account einrichten

1. Account erstellen: https://www.runpod.io/console/signup
2. Credits aufladen: Minimum $10 (reicht für viele Tests!)
3. API Key erstellen:
   - Settings → API Keys → Create API Key
   - Key kopieren (wird nur einmal angezeigt!)

### 2️⃣ Lokales Setup

```bash
cd longcat-video-serverless/

# .env Datei erstellen
cp .env.example .env

# API Key eintragen
nano .env  # oder dein Editor
```

In `.env`:

```bash
RUNPOD_API_KEY=dein-api-key-hier
RUNPOD_ENDPOINT_ID=wird-später-ausgefüllt
```

### 3️⃣ Docker Image bauen

```bash
cd serverless/

# Image bauen (dauert ~10-15 Minuten)
docker build -t longcat-video:latest .

# Testen (lokal, ohne GPU)
docker run --rm longcat-video:latest python -c "import torch; print('OK')"
```

### 4️⃣ Docker Image zu Registry pushen

**Option A: Docker Hub (kostenlos)**

```bash
# Login
docker login

# Tag & Push
docker tag longcat-video:latest DEIN_USERNAME/longcat-video:latest
docker push DEIN_USERNAME/longcat-video:latest
```

**Option B: Runpod Container Registry**

```bash
# Runpod CLI installieren
pip install runpod

# Login
runpod config

# Push
runpod push longcat-video:latest
```

### 5️⃣ Serverless Endpoint erstellen

**Via Runpod Console (einfacher):**

1. https://www.runpod.io/console/serverless
2. **"+ New Endpoint"** klicken
3. Konfiguration:
   ```
   Name: longcat-video
   Image: DEIN_USERNAME/longcat-video:latest
   GPU: RTX 4090 24GB (oder A100)
   Container Disk: 20GB
   Active Workers: 0-1
   Max Workers: 3
   Idle Timeout: 5 seconds
   ```
4. **Deploy** klicken
5. **Endpoint ID** kopieren (z.B. `abc123def456`)

**Via CLI (advanced):**

```bash
runpod serverless create \
  --name longcat-video \
  --image DEIN_USERNAME/longcat-video:latest \
  --gpu RTX4090 \
  --disk-size 20 \
  --min-workers 0 \
  --max-workers 3
```

### 6️⃣ Endpoint ID in .env eintragen

```bash
cd ../
nano .env
```

```bash
RUNPOD_API_KEY=dein-api-key
RUNPOD_ENDPOINT_ID=abc123def456  # ← hier eintragen
```

### 7️⃣ Test Video generieren

```bash
cd local/

# Dependencies installieren
pip install -r requirements.txt

# Test
python client.py \
  --prompt "A cat riding a skateboard through a neon city" \
  --duration 5 \
  --resolution 720p \
  --output test_video.mp4
```

**Erwartete Ausgabe:**

```
🚀 Starte Video-Generierung auf Runpod...
   Task: text_to_video
   Prompt: A cat riding a skateboard through a neon city
   Duration: 5s @ 30fps
✅ Job gestartet: xyz789
⏳ Warte auf Completion...
   Status: IN_PROGRESS (elapsed: 15s)
✨ Video generiert!
   Video URL: https://...
   Execution Time: 23s
⬇️  Downloading video to test_video.mp4...
✅ Video saved to: .../test_video.mp4
```

### 8️⃣ Streamlit App starten (optional)

```bash
streamlit run demo_app.py
```

Browser öffnet sich automatisch → http://localhost:8501

## 💰 Kosten-Optimierung

### Strategie 1: Nur bei Bedarf (günstiger für gelegentliche Nutzung)

```
Active Workers: 0
Max Workers: 1-3
Idle Timeout: 5s
```

- **Kosten**: Nur wenn Video generiert wird
- **Nachteil**: Cold Start ~10-30s

### Strategie 2: Ein Worker warm (schneller)

```
Active Workers: 1
Max Workers: 1
```

- **Kosten**: ~$0.34/h dauerhaft (24h = $8.16/Tag)
- **Vorteil**: Sofortige Antwort, kein Cold Start

### Strategie 3: Hybrid (empfohlen)

```
Active Workers: 0
Max Workers: 3
Warm Start: Manuell bei Bedarf aktivieren
```

- In Runpod Console: Worker manuell warm halten wenn du arbeitest
- Danach deaktivieren

## 🔧 Troubleshooting

### Problem: "Job failed" / GPU Out of Memory

**Lösung**: Größere GPU wählen oder Resolution reduzieren

```bash
# In Runpod Console: Endpoint Settings
GPU: A100 40GB oder A100 80GB
```

### Problem: "Container failed to start"

**Lösung**: Image Logs prüfen

```bash
# Runpod Console → Endpoint → Logs
# Oder via CLI:
runpod logs abc123def456
```

### Problem: Zu langsam / Teuer

**Optimierungen:**

1. Model Quantisierung (INT4 statt FP16)
2. Flash Attention aktivieren
3. Compiled Mode nutzen (`--enable_compile`)
4. Batching mehrerer Requests

## 📊 Benchmark-Referenz

**RTX 4090 (24GB)**:

- 720p, 5s, 30fps: ~20-30s Generation
- 1080p, 5s, 30fps: ~40-60s Generation
- Kosten: ~$0.34/min = **$0.006/s**

**A100 (40GB)**:

- 720p: ~15-20s
- 1080p: ~30-40s
- Kosten: ~$1.10/min = **$0.018/s**

## 🎓 Next Steps

1. **Eigenes Model trainieren**: LongCat-Video fine-tunen
2. **S3 Storage**: Videos persistent speichern
3. **Web UI**: Production-ready Interface
4. **Batch Processing**: Queue-System für viele Videos
5. **Monitoring**: Grafana Dashboard für Costs/Performance

## 🔗 Hilfreiche Links

- Runpod Docs: https://docs.runpod.io
- LongCat-Video: https://huggingface.co/meituan-longcat/LongCat-Video
- Discord Support: https://discord.gg/runpod

## ❓ FAQ

**Q: Kann ich das auch auf AWS/GCP nutzen?**  
A: Ja! Das Docker Image läuft überall. Anpassungen für Lambda/Cloud Run nötig.

**Q: Wie viel VRAM brauche ich wirklich?**  
A: Minimum 24GB (RTX 4090), besser 40GB+ (A100)

**Q: Gibt es günstigere Alternativen zu Runpod?**  
A: Vast.ai (P2P), Modal.com, Replicate

**Q: Kann ich mehrere Videos parallel generieren?**  
A: Ja! Setze `Max Workers > 1`

---

Viel Erfolg! 🚀
