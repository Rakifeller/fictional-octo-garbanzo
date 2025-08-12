import base64
from typing import Any, Dict, List, Optional

import requests
import runpod

from app.pipelines import generate_with_images, load_pipeline
from app.config import USE_LCM
from app.storage import pil_to_base64_png

# Warm-up: load pipeline on cold start
load_pipeline()
print("[serverless] pipeline loaded. USE_LCM:", USE_LCM)

def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

def _read_image_bytes(item: str) -> bytes:
    # If URL -> download; else treat as base64 (dataURL allowed)
    if _is_url(item):
        r = requests.get(item, timeout=60)
        r.raise_for_status()
        return r.content
    # data:image/...;base64,....
    if item.startswith("data:"):
        item = item.split(",", 1)[1]
    return base64.b64decode(item)

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    event.input fields:
      prompt: str (required)
      images: list[str] (each is URL or base64; at least 1 required)
      steps: int (default 28)
      guidance: float (default 5.0)
      seed: int|null
      width: int (default 1024)
      height: int (default 1024)
      return_base64: bool (default True)
    """
    inp = event.get("input") or {}

    prompt: str = inp.get("prompt", "")
    images: List[str] = inp.get("images", [])
    steps: int = int(inp.get("steps", 28))
    guidance: float = float(inp.get("guidance", 5.0))
    seed: Optional[int] = inp.get("seed")
    width: int = int(inp.get("width", 1024))
    height: int = int(inp.get("height", 1024))
    return_base64: bool = bool(inp.get("return_base64", True))

    if not prompt:
        return {"error": "prompt is required"}
    if not images:
        return {"error": "at least one reference image is required in 'images'"}

    try:
        ref_bytes = [_read_image_bytes(x) for x in images]
    except Exception as e:
        return {"error": f"could not read images: {e}"}

    try:
        img = generate_with_images(
            prompt=prompt,
            ref_images=ref_bytes,
            steps=steps,
            guidance=guidance,
            seed=seed,
            size=(width, height)
        )
    except Exception as e:
        return {"error": f"inference failed: {e}"}

    if return_base64:
        return {"image_base64": pil_to_base64_png(img)}

    # For convenience, also return a data URL if caller prefers
    return {"image_data_url": f"data:image/png;base64,{pil_to_base64_png(img)}"}

# Start the serverless worker loop
runpod.serverless.start({"handler": handler})
