# LongCat-Video Serverless Deployment

Hybrid Setup: Lokal entwickeln, GPU-intensive Inference auf Runpod Serverless

## 🎯 Konzept

- **Lokal**: Entwicklung, Testing, UI
- **Runpod Serverless**: GPU-intensive Video-Generierung
- **Kosten**: Nur zahlen wenn Videos generiert werden (pay-per-second)

## 📁 Projekt-Struktur

```
longcat-video-serverless/
├── local/                      # Lokale Entwicklung
│   ├── client.py              # API Client für Runpod
│   ├── demo_app.py            # Streamlit Demo App
│   └── requirements.txt       # Lokale Dependencies
├── serverless/                # Runpod Deployment
│   ├── handler.py             # Serverless Handler
│   ├── Dockerfile             # Container für Runpod
│   └── requirements.txt       # Serverless Dependencies
├── .env.example               # Environment Variables Template
└── README.md
```

## 🚀 Quick Start

### 1. Lokal testen (ohne GPU)

```bash
cd local/
pip install -r requirements.txt
python demo_app.py
```

### 2. Runpod Setup

1. Account erstellen: https://www.runpod.io/
2. API Key holen: Settings → API Keys
3. `.env` erstellen:
   ```
   RUNPOD_API_KEY=your-api-key-here
   ```

### 3. Serverless Endpoint deployen

```bash
cd serverless/
# Docker Image bauen
docker build -t longcat-video:latest .

# Auf Docker Hub pushen (oder Runpod Registry)
docker tag longcat-video:latest YOUR_USERNAME/longcat-video:latest
docker push YOUR_USERNAME/longcat-video:latest
```

4. In Runpod Console:
   - Serverless → New Endpoint
   - Image: `YOUR_USERNAME/longcat-video:latest`
   - GPU: RTX 4090 oder A100
   - Max Workers: 1-3 (je nach Budget)

### 4. Video generieren

```bash
cd local/
python client.py --prompt "A cat riding a skateboard" --duration 5
```

## 💰 Kosten-Kalkulation

**Runpod Serverless (RTX 4090)**:

- Idle: $0.00/min (keine laufenden Kosten!)
- Active: ~$0.34/min (~$0.0057/sec)
- Startup: ~10-30 Sekunden (Cold Start)

**Beispiel**: 10 Videos (je 5 Sekunden, ~30s Generierung):

- Total: ~5min Laufzeit = **~$1.70**

## 🔧 Optimierungen

- **Warm Workers**: 1 Worker warm halten = schnellere Starts (kostet ~$0.34/h)
- **Batching**: Mehrere Videos gleichzeitig generieren
- **Caching**: Häufig genutzte Prompts cachen

## 📊 Alternativen

- **Modal**: Ähnlich wie Runpod, Python-native
- **Replicate**: Managed Deployment (teurer)
- **Banana**: Serverless ML
- **Together.ai**: API-basiert

## 🎥 Features

- ✅ Text-to-Video
- ✅ Image-to-Video
- ✅ Video-Continuation
- ✅ Long-Video Generation (minutes!)
- ✅ Interactive Video

## 📝 Lizenz

MIT (wie LongCat-Video)
# videobla
