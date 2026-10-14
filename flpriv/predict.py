"""Encode one image; optionally show obfuscated (no inverse) version."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flpriv.models.backbone import FaceNet
from flpriv.privacy.transforms import KeyedBlockPerm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--image", required=True)
    ap.add_argument("--obfuscate", action="store_true", help="apply keyed perm without inverse")
    ap.add_argument("--key", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(
        Path(cfg["paths"]["checkpoint_dir"]) / "last.pt",
        map_location=device,
        weights_only=False,
    )
    model = FaceNet(int(ckpt["num_classes"]), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tf = transforms.Compose(
        [
            transforms.Resize((int(cfg["model"].get("image_size", 112)),) * 2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    x = tf(Image.open(args.image).convert("RGB")).unsqueeze(0)
    if args.obfuscate:
        keyed = KeyedBlockPerm(
            grid_g=int(cfg["privacy"].get("grid_g", 4)),
            n_keys=int(cfg["privacy"].get("n_keys", 8)),
            seed=int(cfg["train"].get("seed", 42)),
        )
        x = keyed.apply(x, torch.tensor([args.key]))
    x = x.to(device)
    with torch.no_grad():
        emb = model.embed(x).cpu().numpy()[0].tolist()
        logits = model(x).cpu().numpy()[0].tolist()
    out = {"embedding_dim": len(emb), "embedding": emb, "logits_preview": logits[:10]}
    if args.out:
        Path(args.out).write_text(json.dumps(out) + "\n")
    print(json.dumps({k: out[k] for k in ("embedding_dim", "logits_preview")}))


if __name__ == "__main__":
    main()
