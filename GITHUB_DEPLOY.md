# GitHub → Runpod Serverless Deployment

## 🎯 Einfacherer Weg: Direkt von GitHub deployen!

Runpod kann **direkt aus einem GitHub Repository** deployen - kein Docker Hub nötig!

## 📋 Setup

### 1️⃣ GitHub Repository erstellen

```bash
cd /home/tobi/programmieren/pythonSpielereien/ki/longcat-video-serverless

# Git initialisieren
git init
git add .
git commit -m "Initial commit: LongCat-Video serverless setup"

# GitHub Repo erstellen (via GitHub CLI oder manuell)
gh repo create longcat-video-serverless --public --source=. --push

# Oder manuell auf github.com neues Repo erstellen und:
git remote add origin https://github.com/DEIN_USERNAME/longcat-video-serverless.git
git branch -M main
git push -u origin main
```

### 2️⃣ Runpod Endpoint erstellen (GitHub Integration)

**Option A: Via Runpod Console (empfohlen)**

1. Gehe zu: https://www.runpod.io/console/serverless
2. Klicke **"+ New Endpoint"**
3. Bei **"Container Image"** wähle: **"Build from GitHub"**
4. Konfiguration:

   ```
   Repository: https://github.com/DEIN_USERNAME/longcat-video-serverless
   Branch: main
   Dockerfile Path: serverless/Dockerfile
   Build Context: serverless/

   GPU Type: RTX 4090 24GB
   Container Disk: 20GB

   Workers:
   - Min: 0
   - Max: 3

   Idle Timeout: 5 seconds
   ```

5. **Create Endpoint**

Runpod baut automatisch bei jedem Git-Push ein neues Image!

**Option B: Via runpod.toml Config File**

Erstelle `runpod.toml` im Repository:

```toml
[build]
dockerfile = "serverless/Dockerfile"
context = "serverless/"

[deploy]
gpu = "RTX4090"
container_disk_size_gb = 20
min_workers = 0
max_workers = 3
idle_timeout = 5

[env]
# Environment variables (keine Secrets hier!)
MODEL_PATH = "./weights/LongCat-Video"
```

Dann deployen:

```bash
runpod deploy --config runpod.toml
```

### 3️⃣ Auto-Deploy bei Git Push (CI/CD)

**Option A: Runpod Webhooks**

1. In Runpod Console → Endpoint → Settings
2. Aktiviere **"Auto-rebuild on push"**
3. Kopiere Webhook URL
4. In GitHub Repo → Settings → Webhooks → Add webhook
   - Payload URL: [Runpod Webhook URL]
   - Content type: application/json
   - Events: "Just the push event"

**Option B: GitHub Actions**

Erstelle `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Runpod

on:
  push:
    branches: [main]
    paths:
      - "serverless/**"

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Trigger Runpod Build
        run: |
          curl -X POST ${{ secrets.RUNPOD_WEBHOOK_URL }} \
            -H "Content-Type: application/json" \
            -d '{"ref": "${{ github.ref }}"}'
```

Secrets in GitHub setzen:

- Settings → Secrets → Actions → New secret
- Name: `RUNPOD_WEBHOOK_URL`
- Value: [Webhook URL von Runpod]

### 4️⃣ Development Workflow

```bash
# 1. Lokal entwickeln
cd serverless/
nano handler.py  # Änderungen machen

# 2. Testen (lokal ohne GPU)
python handler.py

# 3. Commit & Push
git add .
git commit -m "Update handler logic"
git push

# 4. Runpod baut automatisch neu! 🎉
# Warte ~5-10 Minuten bis Build fertig ist
```

### 5️⃣ Build Status prüfen

```bash
# Via Runpod CLI
runpod endpoint list
runpod endpoint logs ENDPOINT_ID

# Oder in Console:
# https://www.runpod.io/console/serverless/[ENDPOINT_ID]/builds
```

## 🚀 Vorteile GitHub Integration:

✅ **Kein Docker Hub Account** nötig
✅ **Automatische Builds** bei jedem Push
✅ **Versionskontrolle** integriert
✅ **Einfacher** für Teams
✅ **Kostenlos** (GitHub Public Repos)

## 📝 Beispiel Repository Struktur

```
longcat-video-serverless/
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions (optional)
├── serverless/
│   ├── Dockerfile
│   ├── handler.py
│   └── requirements.txt
├── local/
│   ├── client.py
│   ├── demo_app.py
│   └── requirements.txt
├── runpod.toml                 # Runpod Config (optional)
├── .env.example
├── .gitignore
└── README.md
```

## 💡 Best Practices

1. **Secrets nicht committen!**

   - `.env` in `.gitignore`
   - Secrets als Runpod Environment Variables setzen

2. **Layer Caching nutzen**

   - Dockerfile optimieren
   - Dependencies zuerst, dann Code

3. **Multi-stage Builds** (optional)

   ```dockerfile
   # Build stage
   FROM python:3.10 as builder
   COPY requirements.txt .
   RUN pip install --user -r requirements.txt

   # Runtime stage
   FROM runpod/pytorch:latest
   COPY --from=builder /root/.local /root/.local
   COPY handler.py .
   ```

4. **Branch Strategy**
   - `main` → Production Endpoint
   - `dev` → Development Endpoint
   - Feature branches für Testing

## 🔧 Troubleshooting

**Build schlägt fehl?**
→ Check Runpod Console → Builds → Logs

**Alte Version läuft noch?**
→ Endpoint → Force Rebuild oder neue Version deployen

**Build dauert zu lange?**
→ Layer Caching optimieren, kleineres Base Image

---

**Viel einfacher als Docker Hub, oder? 😉**
