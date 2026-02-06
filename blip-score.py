import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import torch
import torch.nn.functional as F
from PIL import Image

from transformers import Blip2Processor, Blip2ForConditionalGeneration


# -----------------------------
# 1) Config loader / validator
# -----------------------------
def load_blip_prompt_config(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "dimensions" not in cfg or not isinstance(cfg["dimensions"], dict) or len(cfg["dimensions"]) == 0:
        raise ValueError("Config must contain non-empty dict field: 'dimensions'")

    cfg.setdefault("weights", {})
    if not isinstance(cfg["weights"], dict):
        raise ValueError("'weights' must be an object (dict)")

    for dim, item in cfg["dimensions"].items():
        if not isinstance(item, dict):
            raise ValueError(f"dimensions['{dim}'] must be an object")
        for k in ["question", "good_answer", "bad_answer"]:
            if k not in item or not isinstance(item[k], str) or len(item[k].strip()) == 0:
                raise ValueError(f"dimensions['{dim}'].{k} must be a non-empty string")
        ga = item["good_answer"].strip().lower()
        ba = item["bad_answer"].strip().lower()
        if ga not in ["yes", "no"] or ba not in ["yes", "no"] or ga == ba:
            raise ValueError(
                f"dimensions['{dim}'] must have good_answer/bad_answer in {{yes,no}} and different"
            )

    return cfg


# -----------------------------
# 2) BLIP2 yes/no likelihood scoring
# -----------------------------
@dataclass
class YesNoScore:
    p_yes: float
    p_no: float
    loglik_yes: float
    loglik_no: float


class BLIP2CaptionFactorScorer:
    """
    Compute q_caption_factors from BLIP2 by VQA-style yes/no questions.

    For each dimension:
      - Compute log-likelihood of "yes" and "no" answers conditioned on (image, question)
      - Convert to probabilities via softmax
      - dim_score = P(good_answer)
    Then:
      q_caption_factors = weighted average of dim_score
    """

    def __init__(
        self,
        config_json_path: str,
        model_id: str = "Salesforce/blip2-flan-t5-xl",
        device: Optional[str] = None,
        use_fp16: bool = True,
        max_image_size: Optional[int] = None,  # optional: resize long edge to this size
    ):
        self.cfg = load_blip_prompt_config(config_json_path)
        self.dimensions: Dict[str, Dict[str, str]] = self.cfg["dimensions"]
        self.weights: Dict[str, float] = {k: float(v) for k, v in self.cfg.get("weights", {}).items()}

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.use_fp16 = use_fp16 and (self.device.type == "cuda")
        self.max_image_size = max_image_size

        self.processor = Blip2Processor.from_pretrained(model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
        ).to(self.device).eval()

        # Cache tokenized answers
        self._ans_ids = {
            "yes": self.processor.tokenizer("yes", return_tensors="pt", add_special_tokens=False).input_ids,
            "no": self.processor.tokenizer("no", return_tensors="pt", add_special_tokens=False).input_ids,
        }

    def _prep_image(self, image_path: str) -> Image.Image:
        img = Image.open(image_path).convert("RGB")
        if self.max_image_size is not None:
            w, h = img.size
            long_edge = max(w, h)
            if long_edge > self.max_image_size:
                scale = self.max_image_size / float(long_edge)
                nw, nh = int(round(w * scale)), int(round(h * scale))
                img = img.resize((nw, nh), Image.BICUBIC)
        return img

    @torch.no_grad()
    def _loglik_of_answer(self, pil_img: Image.Image, question: str, answer: str) -> float:
        """
        Return total log-likelihood (approx) of the answer tokens given (image, question).
        We compute model loss with labels=answer_ids (mean CE over tokens),
        then convert to total loglik ≈ -loss * num_tokens.
        """
        answer = answer.strip().lower()
        if answer not in ["yes", "no"]:
            raise ValueError("Only yes/no supported in this scorer.")

        inputs = self.processor(images=pil_img, text=question, return_tensors="pt").to(self.device)

        # labels: [1, T]
        labels = self._ans_ids[answer].to(self.device)
        # For seq2seq models, labels should be same batch size. Expand if needed.
        if labels.dim() == 2 and labels.size(0) == 1:
            pass

        # Some models expect padding label = -100; here "yes"/"no" no padding.
        out = self.model(**inputs, labels=labels)
        loss = out.loss  # mean cross-entropy over tokens

        num_tokens = labels.numel()
        loglik = -float(loss.item()) * float(num_tokens)
        return loglik

    @torch.no_grad()
    def score_yes_no(self, pil_img: Image.Image, question: str) -> YesNoScore:
        ll_yes = self._loglik_of_answer(pil_img, question, "yes")
        ll_no = self._loglik_of_answer(pil_img, question, "no")
        logits = torch.tensor([ll_yes, ll_no], dtype=torch.float32)
        probs = F.softmax(logits, dim=0)
        return YesNoScore(
            p_yes=float(probs[0].item()),
            p_no=float(probs[1].item()),
            loglik_yes=float(ll_yes),
            loglik_no=float(ll_no),
        )

    @torch.no_grad()
    def score_image(self, image_path: str) -> Dict[str, Any]:
        pil_img = self._prep_image(image_path)

        dim_scores: Dict[str, float] = {}
        dim_details: Dict[str, Any] = {}

        for dim, item in self.dimensions.items():
            q = item["question"]
            good = item["good_answer"].strip().lower()
            bad = item["bad_answer"].strip().lower()

            yn = self.score_yes_no(pil_img, q)
            good_prob = yn.p_yes if good == "yes" else yn.p_no
            bad_prob = yn.p_yes if bad == "yes" else yn.p_no

            dim_scores[dim] = float(good_prob)
            dim_details[dim] = {
                "question": q,
                "good_answer": good,
                "bad_answer": bad,
                "p_yes": yn.p_yes,
                "p_no": yn.p_no,
                "good_prob": float(good_prob),
                "bad_prob": float(bad_prob),
                "loglik_yes": yn.loglik_yes,
                "loglik_no": yn.loglik_no,
            }

        # weighted average
        wsum = 0.0
        ssum = 0.0
        for dim, s in dim_scores.items():
            w = float(self.weights.get(dim, 1.0))
            wsum += w
            ssum += w * s
        q_caption_factors = ssum / max(1e-12, wsum)

        return {
            "image_path": image_path,
            "model_id": self.model.name_or_path,
            "q_caption_factors": float(q_caption_factors),
            "dim_scores": dim_scores,
            "dim_details": dim_details,
        }

    @torch.no_grad()
    def score_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        return [self.score_image(p) for p in image_paths]


# -----------------------------
# 3) Example usage
# -----------------------------
if __name__ == "__main__":
    cfg_path = "blip_prompts.json"
    img_path = "test.jpg"  # change to your image

    scorer = BLIP2CaptionFactorScorer(
        config_json_path=cfg_path,
        model_id="Salesforce/blip2-flan-t5-xl",  # you can change to smaller/bigger if you have resources
        device=None,
        use_fp16=True,
        max_image_size=672,
    )

    out = scorer.score_image(img_path)
    print("q_caption_factors:", out["q_caption_factors"])
    for k, v in out["dim_scores"].items():
        print(f"{k:16s}: {v:.4f}")

