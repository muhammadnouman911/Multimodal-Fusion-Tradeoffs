import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def _normalize_rel(rel: str) -> str:
    """
    Normalize a path from a CSV cell into a clean, joinable relative path.
    Handles:
      - Leading './' 
      - Mixed forward/back slashes from pandas CSV parsing
      - Escaped backslashes like 'v1\\\\faces\\\\...'
    """
    # Replace escaped backslashes (from CSV parsing)
    rel = rel.replace("\\\\", os.sep).replace("\\", os.sep)
    # Normalize forward slashes to OS sep
    rel = rel.replace("/", os.sep)
    # Strip leading './' or '.\\'
    rel = rel.lstrip(".").lstrip(os.sep)
    return rel


class LoadData(Dataset):
    """
    Titan Loader (v17 — Fixed).
    Supports Multi-Backbone audio features (ECAPA, WavLM, Wav2Vec2).
    Correctly handles:
      - v1_train_English.csv (Train/train(1)/ structure)
      - v3_train_German.csv  (v3/ structure with v3_facenet_feats / v3_ecappa_feats)
      - Submission CSVs      (Dev/val(1)/ structure)
    """
    def __init__(self, csv_paths, feat_dir, schema="train", audio_type="ecappa", lang_label=0):
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]

        dfs = [pd.read_csv(path) for path in csv_paths]
        self.df         = pd.concat(dfs, ignore_index=True)
        self.feat_dir   = feat_dir
        self.schema     = schema
        self.audio_type = audio_type
        self.lang_label = lang_label

        self.face_feats  = []
        self.audio_feats = []
        self.labels      = []

        # ── Feature column map ────────────────────────────────────────────────
        audio_col_map = {
            "ecappa":   "ecappa_feats_path",
            "wavlm":    "wavlmsv_feats_path",
            "wav2vec2": "wav2vec2id_feats_path"
        }
        audio_col = audio_col_map.get(audio_type, "ecappa_feats_path")

        # Detect schema from columns
        has_precomputed = "facenet_feats_path" in self.df.columns
        has_v3          = "face_feat" in self.df.columns
        has_submit      = (not has_precomputed) and ("faces" in self.df.columns)

        # ── Per-row loading ───────────────────────────────────────────────────
        n_skipped = 0
        for _, row in self.df.iterrows():

            if has_precomputed:
                # Standard v1 training CSV: face + audio precomputed paths
                f_rel_raw = str(row["facenet_feats_path"])
                # Audio: try the requested backbone; fallback to ecappa
                a_col = audio_col if audio_col in row.index and pd.notna(row[audio_col]) else "ecappa_feats_path"
                a_rel_raw = str(row[a_col])

                f_rel = _normalize_rel(f_rel_raw)
                a_rel = _normalize_rel(a_rel_raw)

                # Detect v3 German CSV: paths begin with 'facenetfeats/v3...' or './facenetfeats/v3'
                # In v3, the on-disk folder names are 'v3_facenet_feats' and 'v3_ecappa_feats'
                if f_rel.startswith(os.path.join("facenetfeats", "v3")):
                    f_rel = f_rel.replace("facenetfeats" + os.sep + "v3", "v3_facenet_feats" + os.sep + "v3", 1)
                if a_rel.startswith(os.path.join("ecappafeats", "v3")):
                    a_rel = a_rel.replace("ecappafeats" + os.sep + "v3", "v3_ecappa_feats" + os.sep + "v3", 1)

            elif has_v3:
                # v3 CSV schema: face_feat / audio_feat columns
                f_rel = _normalize_rel(str(row.get("face_feat", "")))
                a_rel = _normalize_rel(str(row.get("audio_feat", "")))

            elif has_submit:
                # Submission CSVs: raw face/voice paths, convert extension
                f_rel_raw = str(row["faces"]).replace(".jpg", ".npy").replace(".png", ".npy")
                a_rel_raw = str(row["voices"]).replace(".wav", ".npy")
                if audio_type == "wavlm":
                    a_rel_raw = a_rel_raw.replace("voices", "wavlmsvfeats")
                elif audio_type == "wav2vec2":
                    a_rel_raw = a_rel_raw.replace("voices", "wav2vec2feats")
                # Strip val/ prefix if present
                if f_rel_raw.startswith("val/"):
                    f_rel_raw = f_rel_raw[4:]
                if a_rel_raw.startswith("val/"):
                    a_rel_raw = a_rel_raw[4:]
                f_rel = _normalize_rel(f_rel_raw)
                a_rel = _normalize_rel(a_rel_raw)

            else:
                n_skipped += 1
                continue

            f_path = os.path.join(feat_dir, f_rel)
            a_path = os.path.join(feat_dir, a_rel)

            try:
                face  = np.load(f_path).astype(np.float32)
                if not os.path.exists(a_path):
                    # Fallback: try ecappa path instead of wavlm/wav2vec2
                    a_path_fb = os.path.join(
                        feat_dir,
                        a_rel.replace("wavlmsvfeats", "ecappafeats")
                            .replace("wav2vec2feats", "ecappafeats")
                    )
                    audio = np.load(a_path_fb).astype(np.float32)
                else:
                    audio = np.load(a_path).astype(np.float32)

                label = row.get("label", -1)
                self.face_feats.append(face)
                self.audio_feats.append(audio)
                self.labels.append(label)

            except Exception:
                if schema == "train":
                    n_skipped += 1
                    continue
                else:
                    raise FileNotFoundError(f"Missing feature file:\n  Face:  {f_path}\n  Audio: {a_path}")

        if len(self.face_feats) == 0:
            raise ValueError(
                f"[LoadData] No samples loaded from {csv_paths}.\n"
                f"  feat_dir = {feat_dir}\n"
                f"  schema   = {schema}\n"
                f"  {n_skipped} rows skipped.\n"
                f"  Check that the feature .npy files exist under feat_dir."
            )

        self.face_feats  = np.stack(self.face_feats)
        self.audio_feats = np.stack(self.audio_feats)
        self.labels      = np.array(self.labels, dtype=np.int64)

        if n_skipped > 0:
            print(f"[LoadData] Warning: {n_skipped} rows skipped (missing files) out of {len(self.df)}")

    def __len__(self):
        return len(self.face_feats)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.audio_feats[idx]),
            torch.from_numpy(self.face_feats[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.lang_label, dtype=torch.long)
        )


def load_submit_dataset(config, lang):
    csv_path = config.submit_csv(lang)
    feat_dir = config.submit_feats_dir
    lang_id  = 0 if lang == "English" else 1
    return LoadData(csv_path, feat_dir, schema="submit", audio_type=config.audio_backbone, lang_label=lang_id)
