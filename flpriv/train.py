"""Federated FedAvg training with n-CutMix + keyed perm + inverse decode."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flpriv.data.celeba import CelebAIdentityDataset, resolve_celeba_root, split_clients
from flpriv.models.backbone import FaceNet
from flpriv.privacy.transforms import InverseDecode, KeyedBlockPerm, n_cutmix_batch


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def soft_ce(logits: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    return -(soft * logp).sum(dim=1).mean()


def fedavg(states: List[dict]) -> dict:
    avg = copy.deepcopy(states[0])
    for k in avg:
        if torch.is_floating_point(avg[k]):
            avg[k] = sum(s[k].float() for s in states) / len(states)
            avg[k] = avg[k].to(states[0][k].dtype)
    return avg


def client_train(
    model: FaceNet,
    loader: DataLoader,
    device: torch.device,
    keyed: KeyedBlockPerm,
    decoder: InverseDecode,
    cfg: dict,
    rng: torch.Generator,
) -> dict:
    model.train()
    opt = torch.optim.SGD(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 0.05)),
        momentum=float(cfg["train"].get("momentum", 0.9)),
        weight_decay=float(cfg["train"].get("weight_decay", 5e-4)),
    )
    n_cut = int(cfg["privacy"].get("n_cutmix", 2))
    apply_perm = bool(cfg["privacy"].get("apply_perm", True))
    invert = bool(cfg["privacy"].get("invert_for_train", True))
    n_keys = int(cfg["privacy"].get("n_keys", 8))
    local_epochs = int(cfg["train"].get("local_epochs", 1))
    losses = []
    for _ in range(local_epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            mixed, soft = n_cutmix_batch(
                images,
                labels,
                n_donors=n_cut,
                n_classes=model.classifier.out_features,
                rng=rng,
            )
            soft = soft.to(device)

            key_ids = torch.randint(0, n_keys, (images.size(0),), generator=rng)
            x = mixed
            if apply_perm:
                x = keyed.apply(x.cpu(), key_ids).to(device)
                if invert:
                    x = decoder(x, key_ids.to(device))
            logits = model(x)
            loss = soft_ce(logits, soft)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
    return model.state_dict(), float(np.mean(losses)) if losses else 0.0


def train(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    root = resolve_celeba_root(cfg)
    seed = int(cfg["train"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = torch.Generator()
    rng.manual_seed(seed)

    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None
    image_size = int(cfg["model"].get("image_size", 112))
    train_ds = CelebAIdentityDataset(
        root,
        split="train",
        image_size=image_size,
        train_ratio=float(cfg["data"].get("train_ratio", 0.7)),
        max_n=max_n,
        seed=seed,
        train=True,
    )
    n_clients = int(cfg["train"].get("n_clients", 4))
    clients = split_clients(train_ds, n_clients=n_clients, seed=seed)
    loaders = [
        DataLoader(
            c,
            batch_size=int(cfg["train"].get("batch_size", 64)),
            shuffle=True,
            num_workers=int(cfg["train"].get("num_workers", 4)),
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
        for c in clients
    ]

    model = FaceNet(
        n_classes=train_ds.num_classes,
        embed_dim=int(cfg["model"].get("embed_dim", 512)),
        pretrained=bool(cfg["model"].get("pretrained_backbone", True)),
    ).to(device)
    keyed = KeyedBlockPerm(
        grid_g=int(cfg["privacy"].get("grid_g", 4)),
        n_keys=int(cfg["privacy"].get("n_keys", 8)),
        seed=seed,
    )
    decoder = InverseDecode(keyed)

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rounds = int(cfg["train"].get("rounds", 20))
    history = []
    t0 = time.time()
    for rnd in range(1, rounds + 1):
        states = []
        round_losses = []
        for ci, loader in enumerate(loaders):
            if len(loader) == 0:
                continue
            local = copy.deepcopy(model).to(device)
            state, loss = client_train(local, loader, device, keyed, decoder, cfg, rng)
            states.append(state)
            round_losses.append(loss)
        if not states:
            raise RuntimeError("No client produced updates (empty loaders?)")
        model.load_state_dict(fedavg(states))
        mean_l = float(np.mean(round_losses))
        history.append({"round": rnd, "loss": mean_l, "n_clients": len(states)})
        print(f"[fed] round={rnd}/{rounds} loss={mean_l:.4f} clients={len(states)}")

    elapsed = time.time() - t0
    ckpt = {
        "model": model.state_dict(),
        "num_classes": train_ds.num_classes,
        "config": cfg,
        "keyed_seed": seed,
    }
    torch.save(ckpt, ckpt_dir / "last.pt")
    # also save a no-privacy centralized baseline path marker for eval comparison
    summary = {
        "protocol": cfg["train"].get("protocol", "full"),
        "rounds": rounds,
        "n_clients": n_clients,
        "n_train": len(train_ds),
        "num_classes": train_ds.num_classes,
        "n_cutmix": cfg["privacy"].get("n_cutmix"),
        "grid_g": cfg["privacy"].get("grid_g"),
        "elapsed_sec": elapsed,
        "history": history,
        "checkpoint": str(ckpt_dir / "last.pt"),
    }
    with open(log_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote {log_dir / 'train_summary.json'} elapsed={elapsed:.1f}s")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    train(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
