# stage6_generate/generators.py
import json
import logging
from typing import Iterator

from .errors import GenerationError

log = logging.getLogger("stage6.generators")

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"


class Generator:
    """Interface contract. generate() -> (text, usage); stream() yields text
    deltas and sets self.last_usage when the stream completes."""
    name = "base"

    def generate(self, system: str, user: str,
                 max_tokens: int = 600) -> tuple[str, dict]:
        raise NotImplementedError

    def stream(self, system: str, user: str,
               max_tokens: int = 600) -> Iterator[str]:
        raise NotImplementedError


class OllamaGenerator(Generator):
    """Local generation via Ollama's chat API. `requests` is already
    installed (a flashrank dependency) — zero new packages.

    Defuses the production gotcha: Ollama silently truncates prompts that
    exceed num_ctx FROM THE FRONT — destroying the system prompt and
    top-ranked context first. num_ctx is set explicitly, sized above the
    pipeline's packing budget + history + output. keep_alive=10m avoids
    model reloads between back-to-back queries."""

    def __init__(self, model: str = DEFAULT_MODEL, url: str = DEFAULT_OLLAMA_URL,
                 temperature: float = 0.1, seed: int = 42, num_ctx: int = 8192,
                 timeout: tuple[float, float] = (10.0, 300.0)) -> None:
        self.model = model
        self.name = f"ollama:{model}"
        self.url = url.rstrip("/")
        self.temperature = temperature
        self.seed = seed          # fixed seed => reproducible eval runs (Stage 7)
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.last_usage: dict = {}

    def _options(self, max_tokens: int) -> dict:
        return {"temperature": self.temperature, "seed": self.seed,
                "num_predict": max_tokens, "num_ctx": self.num_ctx}

    def _post(self, payload: dict, stream: bool):
        import requests
        try:
            r = requests.post(f"{self.url}/api/chat", json=payload,
                              timeout=self.timeout, stream=stream)
            r.raise_for_status()
            return r
        except Exception as exc:
            hint = ""
            if "Connection" in type(exc).__name__ or isinstance(exc, OSError):
                hint = (f" — is Ollama running? Start it (`ollama serve` or the "
                        f"desktop app) and check {self.url}")
            raise GenerationError(f"Ollama request failed: {exc}{hint}") from exc

    def generate(self, system: str, user: str,
                 max_tokens: int = 600) -> tuple[str, dict]:
        payload = {"model": self.model, "stream": False,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                   "options": self._options(max_tokens),
                   "keep_alive": "10m"}
        r = self._post(payload, stream=False)
        try:
            data = r.json()
        except ValueError as exc:
            raise GenerationError(f"Ollama returned non-JSON: {exc}") from exc
        text = (data.get("message") or {}).get("content")
        if not text:
            raise GenerationError(f"Ollama returned no content: {str(data)[:200]}")
        usage = {"prompt_tokens": data.get("prompt_eval_count", 0),
                 "completion_tokens": data.get("eval_count", 0)}
        self.last_usage = usage
        return text.strip(), usage

    def stream(self, system: str, user: str,
               max_tokens: int = 600) -> Iterator[str]:
        payload = {"model": self.model, "stream": True,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                   "options": self._options(max_tokens),
                   "keep_alive": "10m"}
        r = self._post(payload, stream=True)
        try:
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                delta = (data.get("message") or {}).get("content") or ""
                if delta:
                    yield delta
                if data.get("done"):
                    self.last_usage = {
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0)}
        finally:
            r.close()


class FakeGenerator(Generator):
    """Deterministic generator for hermetic tests and demos — same interface,
    zero network, zero models."""
    name = "fake"

    def __init__(self, script: str = "The documents state the established fact [1].") -> None:
        self.script = script
        self.calls = 0
        self.last_prompt: tuple[str, str] | None = None
        self.last_usage: dict = {}

    def generate(self, system, user, max_tokens=600):
        self.calls += 1
        self.last_prompt = (system, user)
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        return self.script, dict(self.last_usage)

    def stream(self, system, user, max_tokens=600):
        self.calls += 1
        self.last_prompt = (system, user)
        words = self.script.split(" ")
        for i, w in enumerate(words):
            yield (w + " ") if i < len(words) - 1 else w
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}