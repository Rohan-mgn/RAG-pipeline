import json
import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from stage2_chunk.tokens import TokenEstimator, default_token_estimator

log = logging.getLogger("stage6.session")

_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass
class Turn:
    role: str            # "user" | "assistant"
    text: str
    timestamp: str


class ChatSession:
    """Bounded conversation memory. Token-budgeted (oldest turns dropped
    first), atomically persisted as JSON, or fully in-memory when store_dir
    is None. Memory is injected into the prompt as plain text — the model
    never sees or modifies the storage."""

    def __init__(self, session_id: str | None = None,
                 max_history_tokens: int = 800,
                 store_dir: str | Path | None = "artifacts/sessions",
                 estimator: TokenEstimator = default_token_estimator) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        if not _SESSION_ID.match(self.session_id):
            raise ValueError(
                f"invalid session_id {self.session_id!r}: must match "
                f"{_SESSION_ID.pattern} — it is used verbatim as a filename "
                "and must not contain path separators")
        self.max_history_tokens = max_history_tokens
        self.store_dir = Path(store_dir) if store_dir else None
        self.estimator = estimator
        self.turns: list[Turn] = []
        if self.store_dir:
            self.store_dir.mkdir(parents=True, exist_ok=True)
            self._path = self.store_dir / f"{self.session_id}.json"
            self._load()
        else:
            self._path = None

    def _load(self) -> None:
        if not (self._path and self._path.exists()):
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self.turns = [Turn(**t) for t in data.get("turns", [])]
        except Exception as exc:
            log.warning("corrupt session %s (%s) — starting fresh",
                        self.session_id, exc)
            self.turns = []

    def save(self) -> None:
        if not self._path:
            return
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(json.dumps({"session_id": self.session_id,
                                   "turns": [asdict(t) for t in self.turns]},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    def add(self, role: str, text: str) -> None:
        self.turns.append(Turn(role=role, text=text,
                               timestamp=datetime.now(timezone.utc)
                               .isoformat(timespec="seconds")))
        self._trim()
        self.save()

    def _trim(self) -> None:
        """Drop oldest turns until rendered history fits the budget.
        The newest turn is never dropped."""
        while len(self.turns) > 1 and self._tokens() > self.max_history_tokens:
            dropped = self.turns.pop(0)
            log.debug("session %s: dropped old turn (%d chars)",
                      self.session_id, len(dropped.text))

    def _tokens(self) -> int:
        return self.estimator(self.render() or "")

    def render(self) -> str | None:
        if not self.turns:
            return None
        return "\n".join(f"{'User' if t.role == 'user' else 'Assistant'}: {t.text}"
                         for t in self.turns)

    def clear(self) -> None:
        self.turns = []
        self.save()