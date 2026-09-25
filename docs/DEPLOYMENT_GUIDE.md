# Online Deployment Guide: Efficient Text Recognition System

This guide outlines two straightforward ways to deploy the project online for temporary or permanent demonstration.

---

## ⚡ Method 1: Instant 1-Click Online Demo (Zero Setup, 30 Seconds)

If you need a **live public HTTPS URL right now** to share with evaluators, professors, or team members:

### Option A: Using the 1-Click Batch File
Double-click:
```bash
run_online_demo.bat
```

### Option B: Using the Command Line
```bash
venv\Scripts\python.exe run.py --server --port 8000 --tunnel
```

### What Happens:
1. It launches your local FastAPI web server.
2. It launches an encrypted Cloudflare Quick Tunnel using `cloudflared.exe`.
3. It prints a temporary public HTTPS link in your terminal:
   ```
   ===========================================================================
    [ONLINE] PUBLIC TEMPORARY DEMO URL: https://random-name.trycloudflare.com
   ===========================================================================
   ```
4. The link automatically opens in your browser and is immediately accessible from any computer, smartphone, or tablet worldwide.
5. No account, no API token, and no registration required.
6. When you are done demonstrating, press `Ctrl+C` in the terminal to close the public link.

---

## ☁️ Method 2: Free 24/7 Cloud Deployment on Hugging Face Spaces

If you need the demo to stay online **even when your personal PC is turned off**, Hugging Face Spaces is the recommended platform because it provides a **free 16 GB RAM CPU tier** specifically designed for PyTorch and Transformers apps.

### Steps to Deploy:

1. **Create an Account:**
   - Sign up for free at [huggingface.co](https://huggingface.co).

2. **Create a New Space:**
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
   - Enter Space name: e.g., `efficient-text-recognition`.
   - Select License: `MIT`.
   - Select Space SDK: **Docker** -> **Blank**.
   - Hardware: **CPU Basic (Free - 2 vCPU, 16 GB RAM)**.
   - Visibility: **Public**.
   - Click **Create Space**.

3. **Push Code to Hugging Face:**
   In your project directory terminal:
   ```bash
   # Add Hugging Face as a remote git repository
   git remote add hf https://huggingface.co/spaces/<YOUR_HF_USERNAME>/<SPACE_NAME>

   # Push your code (including Dockerfile and model checkpoints)
   git push hf main
   ```

4. **Your Live Demo:**
   Hugging Face will automatically build the `Dockerfile` and launch your FastAPI Web UI at:
   `https://huggingface.co/spaces/<YOUR_HF_USERNAME>/<SPACE_NAME>`

---

## 🌐 Method 3: Deploying on Render.com

Render offers free web service hosting from GitHub:

1. Push your repository to your GitHub account:
   ```bash
   git add .
   git commit -m "feat: complete OCR web application and deployment config"
   git push origin main
   ```
2. Go to [dashboard.render.com](https://dashboard.render.com) and click **New -> Web Service**.
3. Connect your GitHub repository `pranavakumar01/efficient-text-recognition-system`.
4. Runtime: **Docker**.
5. Instance Type: **Free**.
6. Click **Create Web Service**.
7. Render will build the container using `Dockerfile` and assign a free public `https://<app-name>.onrender.com` URL.
