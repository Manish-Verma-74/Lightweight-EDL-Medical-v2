"""Run a model over a loader and export framework-aware predictions as compressed bundles."""
import json

import numpy as np

KEYS_ALWAYS = ["probs", "native_uncertainty", "msp_uncertainty", "entropy"]
KEYS_OPTIONAL = ["logits", "alpha", "p", "tau", "total_uncertainty"]


def predict_arrays(model, framework, loader, device, non_blocking: bool = True) -> dict:
    """Evaluation-mode predictions for every sample in the loader, in loader order."""
    import torch
    model.eval()
    chunks, labels = {}, []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=non_blocking)
            pred = framework.predict(model(x))
            for k in KEYS_ALWAYS + KEYS_OPTIONAL:
                if k in pred:
                    chunks.setdefault(k, []).append(pred[k].detach().float().cpu().numpy())
            labels.append(np.asarray(y))
    out = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in chunks.items()}
    out["labels"] = np.concatenate(labels).astype(np.int64)
    for k in KEYS_ALWAYS:
        if not np.isfinite(out[k]).all():
            raise ValueError(f"non-finite values in predictions: {k}")
    return out


def save_bundle(path: str, arrays: dict, meta: dict) -> None:
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    payload["meta_json"] = np.array(json.dumps(meta, sort_keys=True, default=str))
    np.savez_compressed(path, **payload)


def load_bundle(path: str):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta_json"}
        meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    return arrays, meta
