import torch
from diffusers import AutoPipelineForText2Image, LCMScheduler
from transformers import CLIPVisionModelWithProjection
from .config import MODEL_ID, USE_LCM
from .storage import bytes_to_pil

_device = "cuda" if torch.cuda.is_available() else "cpu"
_pipe = None
_image_encoder = None

def load_pipeline():
    global _pipe, _image_encoder
    if _pipe is not None:
        return _pipe

    torch.set_grad_enabled(False)

    # CLIP image encoder used by IP-Adapter (ViT-H)
    _image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16
    )

    # SDXL base
    _pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        image_encoder=_image_encoder
    ).to(_device)

    # IP-Adapter Plus Face weights for SDXL ViT-H
    try:
        _pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors"
        )
        _pipe.set_ip_adapter_scale(0.7)
        print("[INFO] IP-Adapter Plus Face loaded and scale set to 0.7")
    except Exception as e:
        print("[WARN] Could not load IP-Adapter Plus Face:", e)

    # Optional speedup: LCM LoRA + LCM Scheduler
    if USE_LCM:
        try:
            _pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
            _pipe.scheduler = LCMScheduler.from_config(_pipe.scheduler.config)
            print("[INFO] LCM enabled")
        except Exception as e:
            print("[WARN] LCM not enabled:", e)

    if _device == "cuda":
        try:
            _pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print("[WARN] xFormers not enabled:", e)

    return _pipe

def generate_with_images(prompt: str, ref_images, steps: int = 28, guidance: float = 5.0, seed=None, size=(1024, 1024)):
    pipe = load_pipeline()
    images = [bytes_to_pil(b) for b in ref_images]
    g = None if seed is None else torch.Generator(device=_device).manual_seed(int(seed))
    result = pipe(
        prompt=prompt,
        ip_adapter_image=images if len(images) > 1 else images[0],
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=g,
        width=int(size[0]),
        height=int(size[1]),
        negative_prompt="deformed, bad anatomy, lowres, text, watermark"
    ).images[0]
    return result
