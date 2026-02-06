import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch
import torch.nn.functional as F
from PIL import Image


# -----------------------------
# 1) Model wrapper (open_clip preferred, fallback to transformers)
# -----------------------------
@dataclass
class ClipBackend:
    name: str  # "open_clip" or "transformers"
    model: object
    preprocess: object
    tokenizer: object


def _try_load_open_clip(
    device: torch.device,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
) -> Optional[ClipBackend]:
    try:
        import open_clip  # type: ignore
    except Exception:
        return None

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return ClipBackend(name="open_clip", model=model, preprocess=preprocess, tokenizer=tokenizer)


def _try_load_transformers_clip(
    device: torch.device,
    model_id: str = "openai/clip-vit-base-patch32",
) -> ClipBackend:
    from transformers import CLIPModel, CLIPProcessor  # type: ignore

    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    return ClipBackend(name="transformers", model=model, preprocess=processor, tokenizer=processor)


# -----------------------------
# 2) Config loader / validator
# -----------------------------
def load_prompt_config(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "dimensions" not in cfg or not isinstance(cfg["dimensions"], dict) or len(cfg["dimensions"]) == 0:
        raise ValueError("Config must contain non-empty dict field: 'dimensions'")

    # validate dimensions structure
    for dim, pn in cfg["dimensions"].items():
        if not isinstance(pn, dict):
            raise ValueError(f"dimensions['{dim}'] must be an object")
        pos = pn.get("pos", None)
        neg = pn.get("neg", None)
        if not isinstance(pos, list) or len(pos) == 0:
            raise ValueError(f"dimensions['{dim}'].pos must be a non-empty list")
        if not isinstance(neg, list) or len(neg) == 0:
            raise ValueError(f"dimensions['{dim}'].neg must be a non-empty list")

    # defaults
    cfg.setdefault("temperature", 1.0)
    cfg.setdefault("weights", {})

    if not isinstance(cfg["temperature"], (int, float)) or cfg["temperature"] <= 0:
        raise ValueError("'temperature' must be a positive number")

    if not isinstance(cfg["weights"], dict):
        raise ValueError("'weights' must be an object (dict)")

    return cfg


# -----------------------------
# 3) CLIP multi-dimension scorer
# -----------------------------
class CLIPMultiDimScorer:
    """
    Compute per-dimension scores using CLIP:
        score_dim = sigmoid( (mean(sim(img, pos_prompts)) - mean(sim(img, neg_prompts))) / temperature )
    Then compute weighted overall score in [0,1].
    """

    def __init__(
        self,
        config_json_path: str,
        device: Optional[str] = None,
        use_fp16: bool = True,
        # backend selection
        prefer_open_clip: bool = True,
        open_clip_model_name: str = "ViT-B-32",
        open_clip_pretrained: str = "openai",
        hf_clip_model_id: str = "openai/clip-vit-base-patch32",
    ):
        self.cfg = load_prompt_config(config_json_path)
        self.temperature = float(self.cfg.get("temperature", 1.0))
        self.prompts: Dict[str, Dict[str, List[str]]] = self.cfg["dimensions"]
        self.weights: Dict[str, float] = {k: float(v) for k, v in self.cfg.get("weights", {}).items()}

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.use_fp16 = use_fp16 and (self.device.type == "cuda")

        backend = None
        if prefer_open_clip:
            backend = _try_load_open_clip(self.device, open_clip_model_name, open_clip_pretrained)

        if backend is None:
            backend = _try_load_transformers_clip(self.device, hf_clip_model_id)

        self.backend = backend

        # Pre-encode all prompts (cached)
        self._text_cache = self._build_text_cache()

    @torch.no_grad()
    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        if self.backend.name == "open_clip":
            tokens = self.backend.tokenizer(texts).to(self.device)
            if self.use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = self.backend.model.encode_text(tokens)
            else:
                feats = self.backend.model.encode_text(tokens)
        else:
            processor = self.backend.tokenizer
            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            if self.use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = self.backend.model.get_text_features(**inputs)
            else:
                feats = self.backend.model.get_text_features(**inputs)

        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def _encode_image(self, pil_img: Image.Image) -> torch.Tensor:
        if self.backend.name == "open_clip":
            x = self.backend.preprocess(pil_img).unsqueeze(0).to(self.device)
            if self.use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = self.backend.model.encode_image(x)
            else:
                feats = self.backend.model.encode_image(x)
        else:
            processor = self.backend.preprocess
            inputs = processor(images=pil_img, return_tensors="pt").to(self.device)
            if self.use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = self.backend.model.get_image_features(**inputs)
            else:
                feats = self.backend.model.get_image_features(**inputs)

        feats = F.normalize(feats, dim=-1)
        return feats.squeeze(0)

    def _build_text_cache(self) -> Dict[str, Dict[str, torch.Tensor]]:
        cache: Dict[str, Dict[str, torch.Tensor]] = {}
        for dim, pn in self.prompts.items():
            pos = pn["pos"]
            neg = pn["neg"]
            cache[dim] = {
                "pos": self._encode_text(pos),  # [P, D]
                "neg": self._encode_text(neg),  # [N, D]
            }
        return cache

    @staticmethod
    def _sigmoid01(x: torch.Tensor) -> float:
        return float(torch.sigmoid(x).clamp(0, 1).item())

    @torch.no_grad()
    def score_image(self, image_path: str) -> Dict[str, Any]:
        pil = Image.open(image_path).convert("RGB")
        img_feat = self._encode_image(pil)  # [D]

        dim_scores: Dict[str, float] = {}
        raw_margins: Dict[str, float] = {}

        for dim, txt in self._text_cache.items():
            pos = txt["pos"]  # [P, D]
            neg = txt["neg"]  # [N, D]
            pos_sim = (pos @ img_feat).mean()
            neg_sim = (neg @ img_feat).mean()
            margin = (pos_sim - neg_sim) / self.temperature
            raw_margins[dim] = float(margin.item())
            dim_scores[dim] = self._sigmoid01(margin)

        # weighted overall score
        wsum = 0.0
        ssum = 0.0
        for dim, s in dim_scores.items():
            w = float(self.weights.get(dim, 1.0))
            wsum += w
            ssum += w * s
        overall = ssum / max(1e-12, wsum)

        return {
            "image_path": image_path,
            "backend": self.backend.name,
            "overall_score": float(overall),
            "dim_scores": dim_scores,
            "raw_margins": raw_margins,
        }

    @torch.no_grad()
    def score_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        return [self.score_image(p) for p in image_paths]


# -----------------------------
# 4) Example usage
# -----------------------------
if __name__ == "__main__":
    # 1) Prepare a config json, e.g. prompts_fiqa.json (as provided previously)
    cfg_path = "prompts_fiqa.json"
    img_path = "test.jpg"  # change to your image

    scorer = CLIPMultiDimScorer(
        config_json_path=cfg_path,
        device=None,                 # auto
        use_fp16=True,
        prefer_open_clip=True,       # use open_clip if available
        open_clip_model_name="ViT-B-32",
        open_clip_pretrained="openai",
        hf_clip_model_id="openai/clip-vit-base-patch32",
    )

    out = scorer.score_image(img_path)
    print("Backend:", out["backend"])
    print("Overall:", out["overall_score"])
    print("Per-dim:")
    for k, v in out["dim_scores"].items():
        print(f"  {k:16s}: {v:.4f}  (margin={out['raw_margins'][k]:+.4f})")

