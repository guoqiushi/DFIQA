"""
quality_classifier.py

Assumptions (per your request):
- clip-score.py and blip-score.py are in the SAME directory as this file.
- config files are in ../config :
    ../config/prompts_fiqa.json
    ../config/blip_prompts.json

Run:
  python quality_classifier.py --img test.jpg
Optional:
  python quality_classifier.py --img test.jpg --alpha 0.7 --device cuda
"""

import os
import json
import argparse
from typing import Any, Dict, Optional, Tuple

import importlib.util


def _import_from_path(py_path: str, symbol: str):
    spec = importlib.util.spec_from_file_location(
        os.path.basename(py_path).replace(".py", ""), py_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import from: {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    if not hasattr(module, symbol):
        raise ImportError(f"Symbol '{symbol}' not found in {py_path}")
    return getattr(module, symbol)


def _safe_get(d: Dict[str, Any], keys: Tuple[str, ...], default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_hard_unusable_rules(
    clip_cfg: Dict[str, Any],
    clip_dim_scores: Dict[str, float],
) -> Optional[str]:
    hard_rules = _safe_get(clip_cfg, ("rules", "hard_unusable"), default=[])
    if not isinstance(hard_rules, list) or len(hard_rules) == 0:
        return None

    def _check_cond(cond: Dict[str, Any]) -> bool:
        dim = cond.get("dim")
        op = cond.get("op")
        val = cond.get("value")
        if not isinstance(dim, str) or dim not in clip_dim_scores:
            return False
        if not isinstance(val, (int, float)) or not isinstance(op, str):
            return False
        x = float(clip_dim_scores[dim])
        v = float(val)
        if op == "<":
            return x < v
        if op == "<=":
            return x <= v
        if op == ">":
            return x > v
        if op == ">=":
            return x >= v
        if op == "==":
            return abs(x - v) < 1e-12
        return False

    for rule in hard_rules:
        if not isinstance(rule, dict):
            continue
        all_conds = rule.get("all", [])
        if not isinstance(all_conds, list) or len(all_conds) == 0:
            continue
        ok = True
        for cond in all_conds:
            if not isinstance(cond, dict) or not _check_cond(cond):
                ok = False
                break
        if ok:
            return str(rule.get("reason", "hard_unusable_rule_triggered"))
    return None


def get_bucket_thresholds(clip_cfg: Dict[str, Any]) -> Tuple[float, float]:
    bt = _safe_get(clip_cfg, ("rules", "bucket_thresholds"), default={})
    if isinstance(bt, dict):
        good_t = bt.get("good", 0.66)
        low_t = bt.get("low", 0.40)
        if isinstance(good_t, (int, float)) and isinstance(low_t, (int, float)):
            return float(good_t), float(low_t)
    return 0.66, 0.40


class QualityClassifier:
    """
    Fuse:
      q_pre = alpha * q_semantic(CLIP overall) + (1-alpha) * q_caption_factors(BLIP2)

    Bucket:
      if hard_unusable -> unusable
      elif q_pre >= good_thr -> good
      elif q_pre >= low_thr  -> poor
      else -> unusable
    """

    def __init__(
        self,
        alpha: float = 0.7,
        device: Optional[str] = None,
        blip_model_id: str = "Salesforce/blip2-flan-t5-xl",
        # paths are auto-resolved based on current file location
        clip_cfg_path: Optional[str] = None,
        blip_cfg_path: Optional[str] = None,
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # same-dir python files
        clip_py = os.path.join(base_dir, "clip-score.py")
        blip_py = os.path.join(base_dir, "blip-score.py")

        if not os.path.isfile(clip_py):
            raise FileNotFoundError(f"Missing {clip_py}")
        if not os.path.isfile(blip_py):
            raise FileNotFoundError(f"Missing {blip_py}")

        # config in ../config
        default_clip_cfg = os.path.normpath(os.path.join(base_dir, "../config/prompts_fiqa.json"))
        default_blip_cfg = os.path.normpath(os.path.join(base_dir, "../config/blip_prompts.json"))

        self.clip_cfg_path = clip_cfg_path or default_clip_cfg
        self.blip_cfg_path = blip_cfg_path or default_blip_cfg

        if not os.path.isfile(self.clip_cfg_path):
            raise FileNotFoundError(f"Missing config: {self.clip_cfg_path}")
        if not os.path.isfile(self.blip_cfg_path):
            raise FileNotFoundError(f"Missing config: {self.blip_cfg_path}")

        CLIPMultiDimScorer = _import_from_path(clip_py, "CLIPMultiDimScorer")
        BLIP2CaptionFactorScorer = _import_from_path(blip_py, "BLIP2CaptionFactorScorer")

        self.clip_cfg = load_json(self.clip_cfg_path)
        self.blip_cfg = load_json(self.blip_cfg_path)

        self.alpha = float(alpha)
        self.good_thr, self.low_thr = get_bucket_thresholds(self.clip_cfg)

        # init scorers
        self.clip_scorer = CLIPMultiDimScorer(
            config_json_path=self.clip_cfg_path,
            device=device,
            use_fp16=True,
            prefer_open_clip=True,
            open_clip_model_name="ViT-B-32",
            open_clip_pretrained="openai",
            hf_clip_model_id="openai/clip-vit-base-patch32",
        )

        self.blip_scorer = BLIP2CaptionFactorScorer(
            config_json_path=self.blip_cfg_path,
            model_id=blip_model_id,
            device=device,
            use_fp16=True,
            max_image_size=672,
        )

    def classify(self, image_path: str) -> Dict[str, Any]:
        clip_out = self.clip_scorer.score_image(image_path)
        blip_out = self.blip_scorer.score_image(image_path)

        q_semantic = float(clip_out["overall_score"])
        q_caption = float(blip_out["q_caption_factors"])
        q_pre = self.alpha * q_semantic + (1.0 - self.alpha) * q_caption

        hard_reason = apply_hard_unusable_rules(self.clip_cfg, clip_out.get("dim_scores", {}))
        if hard_reason is not None:
            label = "unusable"
        else:
            if q_pre >= self.good_thr:
                label = "good"
            elif q_pre >= self.low_thr:
                label = "poor"
            else:
                label = "unusable"

        return {
            "image_path": image_path,
            "label": label,
            "q_pre": float(q_pre),
            "q_semantic_clip": float(q_semantic),
            "q_caption_factors_blip2": float(q_caption),
            "alpha": float(self.alpha),
            "thresholds": {"good": float(self.good_thr), "poor": float(self.low_thr)},
            "hard_unusable_reason": hard_reason,
            "clip_dim_scores": clip_out.get("dim_scores", {}),
            "blip_dim_scores": blip_out.get("dim_scores", {}),
            "config_paths": {
                "clip_cfg": self.clip_cfg_path,
                "blip_cfg": self.blip_cfg_path,
            },
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, required=True, help="input image path")
    ap.add_argument("--alpha", type=float, default=0.7, help="fusion weight for CLIP (0~1)")
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu / None(auto)")
    ap.add_argument("--blip_model", type=str, default="Salesforce/blip2-flan-t5-xl", help="BLIP2 model id")
    ap.add_argument("--clip_cfg", type=str, default=None, help="override clip config path")
    ap.add_argument("--blip_cfg", type=str, default=None, help="override blip config path")
    args = ap.parse_args()

    clf = QualityClassifier(
        alpha=args.alpha,
        device=args.device,
        blip_model_id=args.blip_model,
        clip_cfg_path=args.clip_cfg,
        blip_cfg_path=args.blip_cfg,
    )
    out = clf.classify(args.img)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

