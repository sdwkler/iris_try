# pip install torch
import torch

def list_dinov2_hub_models():
    # 这些入口在 facebookresearch/dinov2 的 hubconf.py 中定义
    candidates = [
        "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14",
        "dinov2_vits14_reg", "dinov2_vitb14_reg", "dinov2_vitl14_reg", "dinov2_vitg14_reg",
        "dinov2_vits14_lc", "dinov2_vitb14_lc", "dinov2_vitl14_lc", "dinov2_vitg14_lc",
    ]
    available = {}
    for name in candidates:
        try:
            torch.hub.load("facebookresearch/dinov2", name, source="github", pretrained=False)
            available[name] = True
        except Exception:
            available[name] = False
    return available

if __name__ == "__main__":
    avail = list_dinov2_hub_models()
    print("DINOv2 (torch.hub):")
    for n, ok in avail.items():
        print(f"{'✓' if ok else '✗'} {n}")