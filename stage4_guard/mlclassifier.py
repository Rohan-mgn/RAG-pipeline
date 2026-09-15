# stage4_guard/mlclassifier.py
import json
import logging
import math
from pathlib import Path

log = logging.getLogger("stage4.ml")

DEFAULT_MODEL = "protectai/deberta-v3-base-prompt-injection"


class MLInjectionClassifier:
    """ONNX prompt-injection classifier loaded directly via onnxruntime +
    tokenizers + huggingface_hub (all already installed as fastembed deps),
    so it works regardless of fastembed's API surface. One-time ~120-200 MB
    download into ./models; ~0.1-0.5 s per query on CPU."""

    name = DEFAULT_MODEL

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: str = "models",
                 max_length: int = 512) -> None:
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        self.max_length = max_length
        self._dir = Path(snapshot_download(
            repo_id=model, cache_dir=cache_dir,
            allow_patterns=["onnx/*", "*.json", "tokenizer.json"]))
        onnx_files = sorted(self._dir.glob("onnx/model*.onnx")) or \
                     sorted(self._dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"no ONNX weights under {self._dir}")
        quantized = [p for p in onnx_files if "quant" in p.name.lower()]
        chosen = (quantized or onnx_files)[0]
        log.info("ml classifier weights: %s", chosen.name)
        self._session = ort.InferenceSession(str(chosen),
                                             providers=["CPUExecutionProvider"])
        self._tok = Tokenizer.from_file(str(self._dir / "tokenizer.json"))
        self._injection_idx = self._resolve_injection_index()

    def _resolve_injection_index(self) -> int:
        try:
            cfg = json.loads((self._dir / "config.json").read_text(encoding="utf-8"))
            id2label = {int(k): str(v) for k, v in cfg.get("id2label", {}).items()}
            for i, label in id2label.items():
                if "inject" in label.lower():
                    return i
        except Exception:
            pass
        return 1        # conventional {0: SAFE, 1: INJECTION}

    def score(self, text: str) -> float | None:
        """P(injection) in [0,1], or None if inference failed (guard then
        leans on the lexical layer — never crashes)."""
        try:
            enc = self._tok.encode(text or "")
            ids = enc.ids[: self.max_length]
            mask = enc.attention_mask[: self.max_length]
            feed = {}
            for inp in self._session.get_inputs():
                if "input_ids" in inp.name:
                    feed[inp.name] = [ids]
                elif "attention" in inp.name:
                    feed[inp.name] = [mask]
                elif "token_type" in inp.name:
                    feed[inp.name] = [enc.type_ids[: self.max_length]]
                else:
                    return None
            logits = self._session.run(None, feed)[0][0]
            m = max(logits)
            exps = [math.exp(v - m) for v in logits]
            return exps[self._injection_idx] / sum(exps)
        except Exception as exc:
            log.warning("ml classifier inference failed: %s", exc)
            return None


def try_load_classifier(model: str = DEFAULT_MODEL, cache_dir: str = "models"):
    """Loaded classifier or None (lexical-only mode). Never raises."""
    try:
        clf = MLInjectionClassifier(model, cache_dir=cache_dir)
        log.info("ml injection classifier loaded: %s", model)
        return clf
    except Exception as exc:
        log.warning("ml classifier unavailable (%s) — lexical-only guard active", exc)
        return None