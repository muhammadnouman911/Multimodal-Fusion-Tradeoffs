import torch


class Evaluator:
    """
    Evaluator with explicit P3 / P4 / P5 / P6 accuracy computation.

    Condition mapping:
        P3 — Audio + Face,    Seen language   (real face)
        P4 — Audio only,      Seen language   (face zeroed)
        P5 — Audio + Face,    Unseen language (real face)
        P6 — Audio only,      Unseen language (face zeroed)

    The mean of P3–P6 is the official competition metric.
    """

    def __init__(self, model, config):
        self.model   = model
        self.config  = config
        self._cached = {}   # id(dataset) → (face, audio, labels) tensors

    # ── Tensor caching ────────────────────────────────────────────────────────

    def _get_tensors(self, dataset):
        key = id(dataset)
        if key not in self._cached:
            self._cached[key] = (
                torch.from_numpy(dataset.face_feats).float(),
                torch.from_numpy(dataset.audio_feats).float(),
                torch.from_numpy(dataset.labels).long(),
            )
        return self._cached[key]

    # ── Core accuracy from raw tensors ────────────────────────────────────────

    def accuracy_from_tensors(
        self,
        face: torch.Tensor,
        audio: torch.Tensor,
        labels: torch.Tensor,
        head: str = "fusion",   # "fusion" | "face" | "audio"
    ) -> float:
        """
        Vectorised accuracy.  Runs the full dataset in one forward pass.

        Args:
            face, audio, labels: GPU/CPU tensors.
            head: which classification head to read from.
        """
        self.model.eval()
        with torch.no_grad():
            out = self.model(face, audio)

            if isinstance(out, dict):
                if head == "fusion":
                    logits = out["fusion_logits"]
                elif head == "face":
                    logits = out["face_logits"]
                elif head == "audio":
                    logits = out["audio_logits"]
                else:
                    raise ValueError(f"Unknown head: {head}")
            else:
                _, logits, _, _ = out

            preds   = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()

        return 100.0 * correct / labels.size(0)

    # ── Dataset-level accuracy (caches tensors) ───────────────────────────────

    def accuracy(self, dataset, head: str = "fusion") -> float:
        face, audio, labels = self._get_tensors(dataset)
        dev = self.config.device
        return self.accuracy_from_tensors(
            face.to(dev),
            audio.to(dev),
            labels.to(dev),
            head=head,
        )

    # ── P3 / P4 / P5 / P6 explicit evaluation ─────────────────────────────────

    def p_accuracy(self, seen_dataset, unseen_dataset) -> dict:
        """
        Compute all four competition conditions and return a dict:
            {
                "P3": float,   "P4": float,
                "P5": float,   "P6": float,
                "mean": float,
            }

        P4 and P6 simulate missing face by zeroing the face tensor.
        """
        dev = self.config.device

        # ── Seen language tensors ─────────────────────────────────────────────
        s_face, s_audio, s_labels = self._get_tensors(seen_dataset)
        s_face   = s_face.to(dev)
        s_audio  = s_audio.to(dev)
        s_labels = s_labels.to(dev)

        # ── Unseen language tensors ───────────────────────────────────────────
        u_face, u_audio, u_labels = self._get_tensors(unseen_dataset)
        u_face   = u_face.to(dev)
        u_audio  = u_audio.to(dev)
        u_labels = u_labels.to(dev)

        # ── Compute each condition ────────────────────────────────────────────
        p3 = self.accuracy_from_tensors(s_face,             s_audio, s_labels)
        p4 = self.accuracy_from_tensors(torch.zeros_like(s_face), s_audio, s_labels)
        p5 = self.accuracy_from_tensors(u_face,             u_audio, u_labels)
        p6 = self.accuracy_from_tensors(torch.zeros_like(u_face), u_audio, u_labels)

        mean_p = (p3 + p4 + p5 + p6) / 4.0

        return {"P3": p3, "P4": p4, "P5": p5, "P6": p6, "mean": mean_p}
