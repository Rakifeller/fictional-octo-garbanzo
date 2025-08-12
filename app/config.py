import os

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_ID = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
USE_LCM = os.getenv("USE_LCM", "false").lower() == "true"
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
API_KEY = os.getenv("API_KEY", "change-me")  # not used in serverless

os.makedirs(DATA_DIR, exist_ok=True)
