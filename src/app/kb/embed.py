"""hotchpotch/bekko-embedding-v1-a8m on ONNX Runtime: 384-d, cosine, multilingual, no prefixes needed.

No torch: just onnxruntime + tokenizers + mean pooling (the model card's recipe), so the image stays small.
"""
import threading

import numpy as np

REPO = "hotchpotch/bekko-embedding-v1-a8m"
DIM = 384
MAX_TOKENS = 512  # model supports 8192; sections are chunked well below 512 tokens


class Embedder:
    def __init__(self) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.enable_padding()
        self.session = ort.InferenceSession(hf_hub_download(REPO, "onnx/model.onnx"), providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.outputs = [o.name for o in self.session.get_outputs()]
        self._lock = threading.Lock()

    def embed(self, texts: list[str], batch: int = 16) -> np.ndarray:
        """L2-normalised embeddings, shape (len(texts), 384)."""
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tokenizer.encode_batch(texts[i : i + batch])
            feed = {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            }
            feed = {k: v for k, v in feed.items() if k in self.inputs}
            with self._lock:
                res = dict(zip(self.outputs, self.session.run(None, feed)))
            if "sentence_embedding" in res:
                vec = res["sentence_embedding"]
            else:  # mean pooling over real tokens
                hidden = res.get("token_embeddings", res.get("last_hidden_state"))
                mask = feed["attention_mask"][..., None].astype(np.float32)
                vec = (hidden * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
            out.append(vec / np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12, None))
        return np.vstack(out).astype(np.float32)


class HashEmbedder:
    """EMBEDDER=fake (CI): words hash into 384 buckets, so texts sharing words are close. No model download."""

    def embed(self, texts: list[str], batch: int = 16) -> np.ndarray:
        import re
        import zlib

        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, zlib.crc32(w.encode()) % DIM] += 1
        return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-9, None)


def get_embedder():
    import os

    return HashEmbedder() if os.getenv("EMBEDDER") == "fake" else Embedder()
