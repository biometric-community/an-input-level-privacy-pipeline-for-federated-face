"""n-CutMix + keyed block permutation + parameter-free inverse decode."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rand_bbox(h: int, w: int, lam: float, rng: torch.Generator) -> Tuple[int, int, int, int]:
    cut_rat = (1.0 - lam) ** 0.5
    cut_h = max(1, int(h * cut_rat))
    cut_w = max(1, int(w * cut_rat))
    cy = int(torch.randint(0, h, (1,), generator=rng).item())
    cx = int(torch.randint(0, w, (1,), generator=rng).item())
    y1 = max(0, cy - cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    y2 = min(h, cy + cut_h // 2)
    x2 = min(w, cx + cut_w // 2)
    return y1, y2, x1, x2


def n_cutmix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    n_donors: int = 2,
    n_classes: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multi-source CutMix → mixed images + soft one-hot targets (B, n_classes)."""
    if rng is None:
        rng = torch.Generator(device="cpu")
    b, c, h, w = images.shape
    mixed = images.clone()
    if n_classes is None:
        n_classes = int(labels.max().item()) + 1
    soft = F.one_hot(labels, num_classes=n_classes).float()
    for _ in range(n_donors):
        idx = torch.randperm(b, generator=rng)
        donor = images[idx]
        donor_y = labels[idx]
        lam = float(torch.empty(1).uniform_(0.3, 0.7).item())
        y1, y2, x1, x2 = _rand_bbox(h, w, lam, rng)
        if y2 <= y1 or x2 <= x1:
            continue
        area = (y2 - y1) * (x2 - x1) / float(h * w)
        mixed[:, :, y1:y2, x1:x2] = donor[:, :, y1:y2, x1:x2]
        soft = soft * (1.0 - area)
        soft = soft + area * F.one_hot(donor_y, num_classes=n_classes).float()
    soft = soft / soft.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return mixed, soft


class KeyedBlockPerm:
    """GxG block permutation codebook; apply / invert by key id."""

    def __init__(self, grid_g: int = 4, n_keys: int = 8, seed: int = 42) -> None:
        self.g = grid_g
        self.l = grid_g * grid_g
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.perms: List[torch.Tensor] = []
        self.invs: List[torch.Tensor] = []
        for _ in range(n_keys):
            p = torch.randperm(self.l, generator=rng)
            inv = torch.empty_like(p)
            inv[p] = torch.arange(self.l)
            self.perms.append(p)
            self.invs.append(inv)

    def _blocks(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,C,H,W → B,L,C,bh,bw
        b, c, h, w = x.shape
        g = self.g
        assert h % g == 0 and w % g == 0
        bh, bw = h // g, w // g
        x = x.view(b, c, g, bh, g, bw).permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(b, self.l, c, bh, bw)

    def _unblocks(self, blocks: torch.Tensor) -> torch.Tensor:
        b, l, c, bh, bw = blocks.shape
        g = self.g
        x = blocks.view(b, g, g, c, bh, bw).permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(b, c, g * bh, g * bw)

    def apply(self, x: torch.Tensor, key_ids: torch.Tensor) -> torch.Tensor:
        blocks = self._blocks(x)
        out = []
        for i in range(x.size(0)):
            p = self.perms[int(key_ids[i].item()) % len(self.perms)]
            out.append(blocks[i, p])
        return self._unblocks(torch.stack(out, 0))

    def invert(self, x: torch.Tensor, key_ids: torch.Tensor) -> torch.Tensor:
        blocks = self._blocks(x)
        out = []
        for i in range(x.size(0)):
            inv = self.invs[int(key_ids[i].item()) % len(self.invs)]
            out.append(blocks[i, inv])
        return self._unblocks(torch.stack(out, 0))


class InverseDecode(nn.Module):
    """Parameter-free module: invert keyed permutation given key ids."""

    def __init__(self, keyed: KeyedBlockPerm) -> None:
        super().__init__()
        self.keyed = keyed

    def forward(self, x: torch.Tensor, key_ids: torch.Tensor) -> torch.Tensor:
        return self.keyed.invert(x, key_ids)
