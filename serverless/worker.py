import base64
from typing import Any, Dict, List, Optional

import requests
import runpod

from app.pipelines import generate_with_images, load_pipeline
from app.storage import pil_to_base64_png

# NOT: burada "load_pipeline()" ÇAĞIRMIYORUZ. (lazy load)

def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

def _read_image_bytes(item: str) -> bytes:
    # URL ise indir; değilse base64 (dataURL destekli)
    if _is_url(item):
        r = requests.get(item, timeout=60)
        r.raise_for_status()
        return r.content
    if item.startswith("data:"):
        item = item.split(",", 1)[1]
    return base64.b64decode(item)

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    event.input:
      prompt: str (required)
      images: list[str] (URL veya base64; >=1)
      steps: int = 20
      guidance: float = 5.0
      seed: int|null
      width: int = 768
      height: int = 768
      return_base64: bool = True
    """
    inp = event.get("input") or {}

    prompt: str = inp.get("prompt", "")
    images: List[str] = inp.get("images", [])
    steps: int = int(inp.get("steps", 20))
    guidance: float = float(inp.get("guidance", 5.0))
    seed: Optional[int] = inp.get("seed")
    width: int = int(inp.get("width", 768))
    height: int = int(inp.get("height", 768))
    return_base64: bool = bool(inp.get("return_base64", True))

    if not prompt:
        return {"error": "prompt is required"}
    if not images:
        return {"error": "at least one reference image is required in 'images'"}

    # LAZY LOAD — ilk istekte modeli yükle ve hata olursa JSON döndür.
    try:
        load_pipeline()
    except Exception as e:
        return {
            "error": "pipeline_load_failed",
            "detail": str(e),
            "hint": "Set HF_TOKEN; optionally mount /models and set HF_HOME=/models"
        }

    # Görselleri oku
    try:
        ref_bytes = [_read_image_bytes(x) for x in images]
    except Exception as e:
        return {"error": "read_images_failed", "detail": str(e)}

    # İnfer
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
        return {"error": "inference_failed", "detail": str(e)}

    if return_base64:
        return {"image_base64": pil_to_base64_png(img)}
    return {"image_data_url": f"data:image/png;base64,{pil_to_base64_png(img)}"}

# Runpod serverless loop
runpod.serverless.start({"handler": handler})
