from typing import Callable

def default_token_estimator(text: str) -> int:
    """~4 chars/token is a robust default for English technical prose.
    Stage 3 injects an exact tiktoken-based estimator at the wiring layer
    when billing-grade counts matter."""
    return max(1, (len(text) + 3) // 4)

TokenEstimator = Callable[[str], int]