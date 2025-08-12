
# AI Selfie – Runpod Serverless Backend

**Purpose:** Generate user‑look‑alike images from selfie(s) using **SDXL + IP‑Adapter Plus Face** on **Runpod Serverless**.

This repo is structured for **Runpod Serverless** (job-based execution). It does **not** open an HTTP port. 
You will call it via Runpod's API (`/v2/<endpointId>/run` or `/runsync`) and receive the result JSON (base64 image).

---

## 🔧 What’s inside?
- `serverless/worker.py` — Runpod handler (`handler(event)`) that loads the model once and serves jobs
- `app/pipelines.py` — Diffusers pipeline (SDXL + IP‑Adapter Face, optional LCM for speed)
- `app/config.py` — Environment config (e.g., `USE_LCM`, `HF_HOME`, `HF_TOKEN` via env)
- `app/storage.py` — Small helpers (bytes→PIL, base64 utils)
- `serverless/requirements.txt` — Minimal deps (PyTorch comes from base image)
- `Dockerfile` — Uses `runpod/pytorch` base image and starts the serverless worker

> If you previously deployed a FastAPI Pod, this repo replaces it with a **pure Serverless** worker.

---

## 🚀 Quick Deploy on Runpod Serverless

1. **Create Endpoint** → *Import Git Repository* (this repo/branch).  
2. **Dockerfile Path:** root (this repo has a single Dockerfile).  
3. **GPU:** L4 / 4090 / A5000 are fine for SDXL.  
4. **Environment Variables (recommended):**
   - `HF_TOKEN=hf_xxx` (needed to pull some models)
   - `USE_LCM=true` (optional, for speed)
   - Optionally cache models on a persistent volume and set `HF_HOME=/models`
5. **(Optional) Persistent Storage:**
   - Mount a volume at `/models` to cache weights between cold starts.
6. **Deploy** — first build may take a bit while models download.

---

## 🧪 How to call the endpoint

### Synchronous (blocking) call
```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync"       -H "Authorization: Bearer <RUNPOD_API_KEY>"       -H "Content-Type: application/json"       -d '{
    "input": {
      "prompt": "studio portrait, soft light, 85mm",
      "images": ["https://example.com/selfie.jpg"],
      "steps": 28,
      "guidance": 5.0,
      "width": 1024,
      "height": 1024
    }
  }'
```

### Asynchronous (recommended)
```bash
# submit job
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/run"       -H "Authorization: Bearer <RUNPOD_API_KEY>"       -H "Content-Type: application/json"       -d '{
    "input": {
      "prompt": "editorial fashion portrait, soft rim light",
      "images": ["data:image/jpeg;base64, ..."]
    }
  }'

# then poll status
curl -H "Authorization: Bearer <RUNPOD_API_KEY>"       "https://api.runpod.ai/v2/<ENDPOINT_ID>/status/<REQUEST_ID>"
```

**Input schema** (JSON):
```json
{
  "prompt": "text prompt (required)",
  "images": ["url-or-base64", "... (>=1 required)"],
  "steps": 28,
  "guidance": 5.0,
  "seed": null,
  "width": 1024,
  "height": 1024,
  "return_base64": true
}
```

**Output**:
```json
{
  "image_base64": "<PNG base64>"
}
```

---

## ⚡ Performance Tips
- Set `USE_LCM=true` (env) to enable **LCM LoRA** + LCM scheduler for 4–8 step fast inference. 
  Quality/texture may slightly differ; tune steps/guidance/prompt.
- Cache models on a Runpod storage volume: mount to `/models` and set `HF_HOME=/models` env.
- For large traffic, increase concurrency by raising min instances or creating multiple endpoints.

---

## 🔒 Security / Limits
- Serverless does not expose HTTP; calls require your **Runpod API key**.
- Enforce your own rate limits in the API layer that calls Runpod.

---

## 📝 License
MIT
