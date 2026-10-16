"""Build the report figures from committed run artifacts (no GPU, no network).

    python scripts/make_figures.py          # writes outputs/figures/*.png

One idea per figure:
  sft_loss.png              SFT validation loss and smoothed training loss
  dpo_val_loss.png          DPO validation loss, HH vs UF
  dpo_val_accuracy.png      DPO validation reward accuracy, HH vs UF
  dpo_val_margin.png        DPO validation reward margin, HH vs UF
  lora_margin.png           persona LoRA runs: held-out margin in log-prob units
  decoding_pilot.png        share of responses hitting max_new_tokens per decoding setting
  judge_dpo_vs_sft.png      win / tie / loss of each DPO model vs SFT
  judge_hh_categories.png   DPO-HH vs SFT win rate per HH category (rep. penalty 1.1)
  persona_style.png, persona_helpfulness.png, persona_safety.png
                            persona LoRA runs vs base SmolLM2, one criterion each (rep. penalty 1.1)
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from alignment.utils import read_json, read_jsonl  # noqa: E402

OUT = ROOT / "outputs" / "figures"

# Validated categorical slots (fixed order) and a blue/grey/red set for win/tie/loss.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
WIN, TIE, LOSS = "#2a78d6", "#c9c8c2", "#e34948"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "axes.labelcolor": INK_2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK_2, "ytick.color": INK_2,
    "legend.frameon": False, "lines.linewidth": 2.0, "axes.axisbelow": True,
})


def ema(values, alpha=0.1):
    """Bias-corrected exponential moving average (no artificial pull towards the first value)."""

    smoothed, average = [], 0.0
    for t, value in enumerate(values, start=1):
        average = alpha * value + (1 - alpha) * average
        smoothed.append(average / (1 - (1 - alpha) ** t))
    return smoothed


def new_figure(height=2.5):
    """Wide, short figures: they render at full page width, so height is what costs report space."""

    return plt.subplots(figsize=(9.0, height))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def end_label(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", fontsize=11, color=color)


# ------------------------------------------------------------------------------------------------
def sft_figure():
    rows = read_jsonl(ROOT / "outputs/sft/metrics.jsonl")
    train = [r for r in rows if r["type"] == "train"]
    evals = [r for r in rows if r["type"] == "eval"]
    fig, ax = new_figure()
    ax.plot([r["step"] for r in train], ema([r["train_loss"] for r in train]), color=BLUE, label="training (smoothed)")
    ax.plot([r["step"] for r in evals], [r["val_loss"] for r in evals], color=ORANGE, marker="o", markersize=5,
            label="validation")
    ax.annotate(f"{evals[0]['val_loss']:.3f}", (0, evals[0]["val_loss"]), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9, color=INK_2)
    ax.annotate(f"{evals[-1]['val_loss']:.3f}", (evals[-1]["step"], evals[-1]["val_loss"]), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=9, color=INK_2)
    ax.set(xlabel="optimizer step", ylabel="loss (NLL per response token)",
           title=f"SFT: validation loss falls from {evals[0]['val_loss']:.3f} to {evals[-1]['val_loss']:.3f}")
    ax.legend(loc="lower left")
    save(fig, "sft_loss.png")


def dpo_evals():
    data = {}
    for track in ["hh", "uf"]:
        data[track] = [r for r in read_jsonl(ROOT / f"outputs/dpo_{track}/metrics.jsonl") if r["type"] == "eval"]
    return data


def dpo_figure(key, name, ylabel, title, skip_step0=False):
    data = dpo_evals()
    fig, ax = new_figure()
    for track, label, color in [("uf", "UltraFeedback", ORANGE), ("hh", "HH-RLHF", BLUE)]:
        rows = [r for r in data[track] if not (skip_step0 and r["step"] == 0)]
        steps, values = [r["step"] for r in rows], [r[key] for r in rows]
        ax.plot(steps, values, color=color, marker="o", markersize=4, label=label)
        end_label(ax, steps[-1], values[-1], label, color)
    ax.set(xlabel="optimizer step", ylabel=ylabel, title=title)
    ax.set_xlim(right=ax.get_xlim()[1] + 90)  # room for the end labels
    return fig, ax


def dpo_figures():
    fig, ax = dpo_figure("val_loss", "dpo_val_loss.png", "validation DPO loss",
                         "DPO validation loss (both start at ln 2 = 0.693)")
    save(fig, "dpo_val_loss.png")

    best = {t: read_json(ROOT / f"outputs/dpo_{t}/summary.json") for t in ["hh", "uf"]}
    fig, ax = dpo_figure("val_reward_accuracy", "dpo_val_accuracy.png", "validation reward accuracy",
                         f"DPO reward accuracy: best {best['uf']['best_val_reward_accuracy']:.2f} (UF), "
                         f"{best['hh']['best_val_reward_accuracy']:.2f} (HH)", skip_step0=True)
    ax.axhline(0.5, color=INK_2, linewidth=0.8, linestyle=":")
    ax.annotate("chance", (ax.get_xlim()[0], 0.5), xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_ylim(0.45, 0.8)
    save(fig, "dpo_val_accuracy.png")

    fig, ax = dpo_figure("val_reward_margin", "dpo_val_margin.png", "validation reward margin",
                         "DPO reward margin keeps growing through training")
    save(fig, "dpo_val_margin.png")


def lora_figure():
    runs = [("lora_dpo", "3 epochs, beta 0.1", BLUE), ("lora_dpo_beta05", "3 epochs, beta 0.5", AQUA),
            ("lora_dpo_1ep", "1 epoch, beta 0.1", ORANGE)]
    fig, ax = new_figure()
    for run, label, color in runs:
        beta = read_json(ROOT / f"outputs/{run}/config.json")["beta"]
        rows = read_jsonl(ROOT / f"outputs/{run}/metrics.jsonl")
        evals = [({**r, "step": 0} if r.get("type") == "initial_eval" else r) for r in rows if "eval_loss" in r]
        evals = sorted({r["step"]: r for r in evals}.values(), key=lambda r: r["step"])
        steps = [r["step"] for r in evals]
        values = [r["eval_rewards/margins"] / beta for r in evals]
        ax.plot(steps, values, color=color, marker="o", markersize=4)
        end_label(ax, steps[-1], values[-1], label, color)
    ax.set_xlim(right=ax.get_xlim()[1] + 28)
    ax.set(xlabel="optimizer step", ylabel="held-out margin / beta (log-prob)",
           title="Persona LoRA: the 3-epoch run moves furthest from the base model")
    save(fig, "lora_margin.png")


def pilot_figure():
    summary = read_json(ROOT / "outputs/pilot/summary.json")
    settings = [("greedy", "greedy"), ("rp1.1", "penalty 1.1"), ("rp1.2", "penalty 1.2"), ("ngram4", "no-repeat\n4-gram")]
    models = [("sft", "SFT", BLUE), ("dpo_uf", "DPO-UF", ORANGE), ("dpo_hh", "DPO-HH", AQUA)]
    fig, ax = new_figure()
    width = 0.26
    for offset, (model, label, color) in enumerate(models):
        xs = [i + (offset - 1) * (width + 0.02) for i in range(len(settings))]
        values = [100 * summary[f"{model}__{s}"]["truncated_rate"] for s, _ in settings]
        ax.bar(xs, values, width=width, color=color, label=label)
        for x, value in zip(xs, values):
            if value == 0:
                ax.annotate("0", (x, 0), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(settings)), [label for _, label in settings])
    ax.set(ylabel="% of answers that never stop", title="Repetition penalty 1.1 stops most loops")
    ax.legend(loc="upper right")
    save(fig, "decoding_pilot.png")


def stacked_rows(ax, rows):
    """Horizontal 100% bars: (label, wins, ties, losses)."""

    for y, (label, wins, ties, losses) in enumerate(rows):
        total, left = wins + ties + losses, 0.0
        for count, color in [(wins, WIN), (ties, TIE), (losses, LOSS)]:
            share = 100 * count / total
            ax.barh(y, share, left=left, height=0.6, color=color, edgecolor=SURFACE, linewidth=2)
            if share >= 7:
                ax.text(left + share / 2, y, str(count), ha="center", va="center", fontsize=11,
                        color="white" if color != TIE else INK)
            left += share
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.axvline(50, color=INK_2, linewidth=0.8, linestyle=":", zorder=0)
    ax.grid(axis="y", visible=False)


def legend_below(ax, labels):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (WIN, TIE, LOSS)]
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1)


def judge_dpo_figure():
    rows = []
    for track, label in [("dpo_uf_vs_sft_alpaca", "UF vs SFT"), ("dpo_hh_vs_sft_hh", "HH vs SFT")]:
        for tag, tag_label in [("rp1.1", "penalty 1.1"), ("greedy", "greedy")]:
            c = read_json(ROOT / f"outputs/judged/{tag}/{track}_judged_summary.json")["criteria"]["overall"]
            rows.append((f"{label} ({tag_label})", c["wins"], c["ties"], c["losses"]))
    fig, ax = new_figure(2.6)
    stacked_rows(ax, rows)
    ax.set_xlabel("% of 300 prompts")
    ax.set_title("Neither DPO model beats SFT")
    legend_below(ax, ["DPO better", "tie", "SFT better"])
    save(fig, "judge_dpo_vs_sft.png")


def judge_categories_figure():
    counts = collections.defaultdict(collections.Counter)
    for r in read_jsonl(ROOT / "outputs/judged/rp1.1/dpo_hh_vs_sft_hh_judged.jsonl"):
        counts[r["category"]][r["outcomes"]["overall"]] += 1
    rates = sorted(((100 * v["win"] / max(1, v["win"] + v["loss"]), c) for c, v in counts.items()), reverse=True)
    fig, ax = new_figure(3.4)
    for y, (rate, category) in enumerate(rates):
        ax.barh(y, rate, height=0.6, color=WIN if rate > 50 else LOSS)
        ax.annotate(f"{rate:.0f}%", (rate, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=11, color=INK_2)
    ax.set_yticks(range(len(rates)), [c.replace("_", " ") for _, c in rates])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.axvline(50, color=INK_2, linewidth=0.8, linestyle=":")
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="DPO-HH win rate vs SFT, ties excluded (~30 prompts each)",
           title="DPO-HH vs SFT by category: wins on hate, cyber abuse, harmful requests")
    save(fig, "judge_hh_categories.png")


def persona_figures():
    runs = [("lora_dpo", "3 epochs, beta 0.1"), ("lora_dpo_1ep", "1 epoch, beta 0.1"), ("lora_dpo_beta05", "3 epochs, beta 0.5")]
    titles = {"style": "Persona style: only the 3-epoch run changed",
              "helpfulness": "Helpfulness vs base model",
              "safety": "Safety vs base model: mostly ties"}
    for criterion, title in titles.items():
        rows = []
        for run, label in runs:
            c = read_json(ROOT / f"outputs/judged/rp1.1/{run}_vs_smollm_base_persona_judged_summary.json")["criteria"][criterion]
            rows.append((label, c["wins"], c["ties"], c["losses"]))
        fig, ax = new_figure(2.0)
        stacked_rows(ax, rows)
        ax.set_xlabel("% of 24 prompts")
        ax.set_title(title)
        legend_below(ax, ["LoRA model better", "tie", "base model better"])
        save(fig, f"persona_{criterion}.png")


if __name__ == "__main__":
    for old in ["sft_training.png", "dpo_training.png", "lora_training.png", "judge_persona.png"]:
        (OUT / old).unlink(missing_ok=True)
    sft_figure()
    dpo_figures()
    lora_figure()
    pilot_figure()
    judge_dpo_figure()
    judge_categories_figure()
    persona_figures()
