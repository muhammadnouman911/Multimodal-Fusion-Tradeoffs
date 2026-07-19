class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Args:
        patience:  Number of epochs to wait after last improvement.
        min_delta: Minimum absolute improvement to be counted.
        mode:      "max" (higher is better) or "min" (lower is better).
    """

    def __init__(
        self,
        patience: int   = 15,
        min_delta: float = 0.1,
        mode: str        = "max",
    ):
        assert mode in ("max", "min"), f"mode must be 'max' or 'min', got {mode}"
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best_score = None
        self.counter    = 0

    def step(self, score: float) -> bool:
        """
        Call once per epoch.

        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
