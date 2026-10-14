"""Build SVG figures + REPORT.md from real train/eval JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flpriv.plot_style import FIGSIZE, LINEWIDTH_MAIN, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def plot_train(history, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(
        [h["round"] for h in history],
        [h["loss"] for h in history],
        color=MULTI_SERIES_COLORS[0],
        linewidth=LINEWIDTH_MAIN,
    )
    ax.set_xlabel("FedAvg round")
    ax.set_ylabel("Client mean loss")
    ax.set_title("Federated training loss")
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_privacy(eval_res: dict, out: Path) -> None:
    apply_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    obf = eval_res.get("obfuscation", {})
    axes[0].bar(
        [0, 1],
        [obf.get("clean_top1", 0) * 100, obf.get("obfuscated_top1", 0) * 100],
        color=MULTI_SERIES_COLORS[:2],
        edgecolor="#002080",
    )
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Clean", "Obfuscated"])
    axes[0].set_ylabel("Top-1 (%)")
    axes[0].set_title("Authorized vs unauthorized view")
    apply_style(axes[0], grid_axis="y")
    axes[1].bar([0], [eval_res.get("mia_auc", 0)], color=MULTI_SERIES_COLORS[0], edgecolor="#002080")
    axes[1].axhline(0.5, color=MULTI_SERIES_COLORS[1], linewidth=2, linestyle="--", label="chance")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(["MIA AUC"])
    axes[1].set_title("Membership inference (max-softmax)")
    axes[1].legend()
    apply_style(axes[1], grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def write_report(cfg, train_s, eval_res, path: Path) -> None:
    protocol = cfg["train"].get("protocol", "full")
    lfw = eval_res.get("lfw", {})
    obf = eval_res.get("obfuscation", {})
    lines = [
        f"# REPORT: Federated Input-Level Privacy — {protocol}",
        "",
        f"**Protocol:** `{protocol}`  ",
        f"**Package:** `flpriv`  ",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| LFW accuracy | {lfw.get('lfw_accuracy', float('nan'))*100:.2f}% |",
        f"| LFW EER | {lfw.get('lfw_eer', float('nan'))*100:.2f}% |",
        f"| MIA AUC (max-softmax) | {eval_res.get('mia_auc', float('nan')):.4f} |",
        f"| Clean top-1 | {obf.get('clean_top1', float('nan'))*100:.2f}% |",
        f"| Obfuscated top-1 (no inverse) | {obf.get('obfuscated_top1', float('nan'))*100:.2f}% |",
        "",
        "## Training",
        "",
        f"- Rounds: {train_s.get('rounds')}",
        f"- Clients: {train_s.get('n_clients')}",
        f"- Train images: {train_s.get('n_train')}",
        f"- Identities: {train_s.get('num_classes')}",
        f"- n-CutMix / grid G: {train_s.get('n_cutmix')} / {train_s.get('grid_g')}",
        f"- Elapsed: {train_s.get('elapsed_sec', 0):.2f} s",
        f"- Checkpoint: `{train_s.get('checkpoint')}`",
        "",
        "## Figures",
        "",
        "### `train_loss.svg`",
        "",
        "![train_loss](figures/train_loss.svg)",
        "",
        "### `privacy_metrics.svg`",
        "",
        "![privacy_metrics](figures/privacy_metrics.svg)",
        "",
        "## Reproduce",
        "",
        "```bash",
        "bash scripts/smoke.sh",
        "bash scripts/train_full.sh",
        "```",
        "",
        "## Notes",
        "",
        "- CelebA partitioned by identity for FedAvg; LFW for verification.",
        "- PubFig/PINS omitted (see DEVIATIONS.md).",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    train_s = json.loads((log_dir / "train_summary.json").read_text())
    eval_res = json.loads((log_dir / "eval_results.json").read_text())
    plot_train(train_s.get("history", []), fig_dir / "train_loss.svg")
    plot_privacy(eval_res, fig_dir / "privacy_metrics.svg")
    tracked = _ROOT / "figures"
    tracked.mkdir(exist_ok=True)
    for name in ("train_loss.svg", "privacy_metrics.svg"):
        src = fig_dir / name
        if src.exists():
            (tracked / name).write_bytes(src.read_bytes())
    write_report(cfg, train_s, eval_res, _ROOT / "REPORT.md")
    print(f"Wrote {_ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
