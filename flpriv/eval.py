"""LFW verification + membership-inference proxy + obfuscation check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flpriv.data.celeba import CelebAIdentityDataset, resolve_celeba_root
from flpriv.data.lfw import LFWPairsDataset, resolve_lfw_root
from flpriv.models.backbone import FaceNet
from flpriv.privacy.transforms import KeyedBlockPerm, n_cutmix_batch


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def best_acc_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_acc, best_eer = 0.0, 1.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int32)
        best_acc = max(best_acc, float((pred == labels).mean()))
        pos, neg = labels == 1, labels == 0
        if pos.any() and neg.any():
            fpr = float((pred[neg] == 1).mean())
            fnr = float((pred[pos] == 0).mean())
            best_eer = min(best_eer, 0.5 * (fpr + fnr))
    return best_acc, best_eer


@torch.no_grad()
def lfw_eval(model: FaceNet, loader, device) -> Dict:
    scores, labels = [], []
    model.eval()
    for a, b, y in tqdm(loader, desc="lfw", leave=False):
        ea = model.embed(a.to(device))
        eb = model.embed(b.to(device))
        scores.append((ea * eb).sum(dim=1).cpu().numpy())
        labels.append(y.numpy())
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    acc, eer = best_acc_eer(s, y)
    return {"lfw_accuracy": acc, "lfw_eer": eer, "n_pairs": int(len(y))}


@torch.no_grad()
def membership_inference_auc(
    model: FaceNet,
    member_loader,
    nonmember_loader,
    device,
) -> float:
    """Oriented max-softmax MIA proxy: higher max softmax → more member-like."""
    model.eval()
    scores, labels = [], []
    for loader, lab in ((member_loader, 1), (nonmember_loader, 0)):
        for images, _y in tqdm(loader, desc=f"mia-{lab}", leave=False):
            logits = model(images.to(device))
            conf = F.softmax(logits, dim=1).max(dim=1).values.cpu().numpy()
            scores.append(conf)
            labels.append(np.full(len(conf), lab))
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


@torch.no_grad()
def obfuscation_identity_drop(
    model: FaceNet,
    loader,
    keyed: KeyedBlockPerm,
    device,
    n_cut: int,
    n_keys: int,
    seed: int,
) -> Dict:
    """Compare top-1 on clean vs n-CutMix+perm without inverse (unauthorized view)."""
    model.eval()
    rng = torch.Generator()
    rng.manual_seed(seed)
    correct_clean = correct_obf = total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = model(images).argmax(dim=1)
        correct_clean += int((pred == labels).sum())
        mixed, _ = n_cutmix_batch(
            images,
            labels,
            n_donors=n_cut,
            n_classes=model.classifier.out_features,
            rng=rng,
        )
        key_ids = torch.randint(0, n_keys, (images.size(0),), generator=rng)
        obf = keyed.apply(mixed.cpu(), key_ids).to(device)
        pred_o = model(obf).argmax(dim=1)
        correct_obf += int((pred_o == labels).sum())
        total += images.size(0)
    return {
        "clean_top1": correct_clean / max(total, 1),
        "obfuscated_top1": correct_obf / max(total, 1),
        "n": total,
    }


def evaluate(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    celeba = resolve_celeba_root(cfg)
    lfw = resolve_lfw_root(cfg)
    seed = int(cfg["train"].get("seed", 42))
    image_size = int(cfg["model"].get("image_size", 112))
    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None

    ckpt_path = Path(cfg["paths"]["checkpoint_dir"]) / "last.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FaceNet(
        n_classes=int(ckpt["num_classes"]),
        embed_dim=int(cfg["model"].get("embed_dim", 512)),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    max_pairs = cfg["eval"].get("max_lfw_pairs")
    max_pairs = int(max_pairs) if max_pairs is not None else None
    lfw_ds = LFWPairsDataset(lfw, image_size=image_size, max_pairs=max_pairs)
    lfw_loader = DataLoader(
        lfw_ds,
        batch_size=int(cfg["eval"].get("batch_size", 128)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 4)),
    )
    results = {"lfw": lfw_eval(model, lfw_loader, device)}
    print(f"[lfw] acc={results['lfw']['lfw_accuracy']:.4f} eer={results['lfw']['lfw_eer']:.4f}")

    mia_max = cfg["eval"].get("mia_max")
    mia_max = int(mia_max) if mia_max is not None else None
    if max_n is not None and mia_max is not None:
        mia_max = min(mia_max, max_n)
    member = CelebAIdentityDataset(
        celeba, split="train", image_size=image_size, max_n=mia_max, seed=seed, train=False
    )
    nonmember = CelebAIdentityDataset(
        celeba, split="val", image_size=image_size, max_n=mia_max, seed=seed, train=False
    )
    # align class count by remapping not needed — MIA uses max softmax only
    # but classifier expects train num_classes; val labels may exceed — use dummy forward images only
    mem_loader = DataLoader(member, batch_size=int(cfg["eval"].get("batch_size", 128)), shuffle=False, num_workers=2)
    non_loader = DataLoader(nonmember, batch_size=int(cfg["eval"].get("batch_size", 128)), shuffle=False, num_workers=2)
    mia = membership_inference_auc(model, mem_loader, non_loader, device)
    results["mia_auc"] = mia
    print(f"[mia] max-softmax AUC={mia:.4f}")

    keyed = KeyedBlockPerm(
        grid_g=int(cfg["privacy"].get("grid_g", 4)),
        n_keys=int(cfg["privacy"].get("n_keys", 8)),
        seed=seed,
    )
    probe = CelebAIdentityDataset(
        celeba, split="val", image_size=image_size, max_n=mia_max, seed=seed, train=False
    )
    # Remap probe labels into [0, num_classes) randomly for top1 probe on train head — use train set instead
    probe = CelebAIdentityDataset(
        celeba, split="train", image_size=image_size, max_n=mia_max, seed=seed, train=False
    )
    probe_loader = DataLoader(probe, batch_size=64, shuffle=False, num_workers=2)
    obf = obfuscation_identity_drop(
        model,
        probe_loader,
        keyed,
        device,
        n_cut=int(cfg["privacy"].get("n_cutmix", 2)),
        n_keys=int(cfg["privacy"].get("n_keys", 8)),
        seed=seed,
    )
    results["obfuscation"] = obf
    print(
        f"[obf] clean_top1={obf['clean_top1']:.4f} obfuscated_top1={obf['obfuscated_top1']:.4f}"
    )

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"Wrote {out}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
