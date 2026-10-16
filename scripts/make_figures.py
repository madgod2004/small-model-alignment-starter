"""Build the report figures from committed run artifacts (no GPU, no network).

    python scripts/make_figures.py          # writes outputs/figures/*.png

Figures:
  sft_training.png        SFT train loss (raw + EMA) and validation loss, best checkpoint marked
  dpo_training.png        DPO loss / reward accuracy / reward margin for HH-RLHF and UltraFeedback
  lora_training.png       LoRA-DPO held-out loss / accuracy / margin (log-prob units) for the 3 runs
  decoding_pilot.png      share of responses hitting max_new_tokens per decoding setting
  judge_dpo_vs_sft.png    win / tie / loss of each DPO model vs SFT (greedy and rp1.1)
  judge_hh_categories.png DPO-HH vs SFT win rate per HH category
  judge_persona.png       LoRA-DPO runs vs base SmolLM2 per criterion
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from alignment.utils import read_json, read_jsonl  # noqa: E402

OUT = ROOT / "outputs" / "figures"

# Validated categorical slots (fixed order) and a blue/grey/red diverging set for win/tie/loss.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
WIN, TIE, LOSS = "#2a78d6", "#c9c8c2", "#e34948"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.labelcolor": INK_2,
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


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def mark_best(ax, step, label="best ckpt"):
    ax.axvline(step, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
    ax.annotate(f"{label} (step {step})", (step, 1), xycoords=("data", "axes fraction"), xytext=(-4, -4),
                textcoords="offset points", ha="right", va="top", fontsize=8, color=INK_2)


# ------------------------------------------------------------------------------------------------
def sft_figure():
    rows = read_jsonl(ROOT / "outputs/sft/metrics.jsonl")
    train = [r for r in rows if r["type"] == "train"]
    evals = [r for r in rows if r["type"] == "eval"]
    best = read_json(ROOT / "outputs/sft/summary.json")["best_step"]
    steps = [r["step"] for r in train]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(steps, [r["train_loss"] for r in train], color=BLUE, linewidth=0.8, alpha=0.35, label="train (per logged step)")
    ax.plot(steps, ema([r["train_loss"] for r in train]), color=BLUE, label="train (bias-corrected EMA, α=0.1)")
    ax.plot([r["step"] for r in evals], [r["val_loss"] for r in evals], color=ORANGE, marker="o", markersize=5,
            label="validation (1,000 ex.)")
    ax.annotate(f"{evals[0]['val_loss']:.3f} (base model)", (0, evals[0]["val_loss"]), xytext=(6, 6),
                textcoords="offset points", fontsize=8, color=INK_2)
    ax.annotate(f"{evals[-1]['val_loss']:.3f}", (evals[-1]["step"], evals[-1]["val_loss"]), xytext=(-4, -12),
                textcoords="offset points", fontsize=8, color=INK_2, ha="right")
    mark_best(ax, best)
    ax.set(xlabel="optimizer step (effective batch 32)", ylabel="response-token NLL",
           title="SFT on Qwen2.5-0.5B (30k UltraChat examples, 1 epoch)")
    ax.legend(loc="lower left")
    save(fig, "sft_training.png")


def dpo_figure():
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True)
    panels = [("loss", "DPO loss"), ("reward_accuracy", "reward accuracy"), ("reward_margin", "reward margin (β·Δ log-prob)")]
    for row, (track, title) in enumerate([("hh", "HH-RLHF"), ("uf", "UltraFeedback")]):
        rows = read_jsonl(ROOT / f"outputs/dpo_{track}/metrics.jsonl")
        train = [r for r in rows if r["type"] == "train"]
        evals = [r for r in rows if r["type"] == "eval"]
        best = read_json(ROOT / f"outputs/dpo_{track}/summary.json")["best_step"]
        for col, (key, label) in enumerate(panels):
            ax = axes[row, col]
            values = [r[f"train_{key}"] for r in train]
            ax.plot([r["step"] for r in train], values, color=BLUE, linewidth=0.8, alpha=0.3)
            ax.plot([r["step"] for r in train], ema(values), color=BLUE, label="train (bias-corrected EMA, α=0.1)")
            ax.plot([r["step"] for r in evals], [r[f"val_{key}"] for r in evals], color=ORANGE, marker="o", markersize=4,
                    label="validation (1,000 pairs)")
            if key == "loss":
                ax.axhline(0.6931, color=INK_2, linewidth=0.8, linestyle=":")
                ax.annotate("ln 2 (policy = reference)", (0, 0.6931), xytext=(4, 3), textcoords="offset points",
                            fontsize=7, color=INK_2)
            if key == "reward_accuracy":
                ax.axhline(0.5, color=INK_2, linewidth=0.8, linestyle=":")
                ax.set_ylim(0, 1)
                ax.annotate("step 0: policy = reference,\nall rewards tied -> 0 (strict >)", (0, 0), xytext=(8, 6),
                            textcoords="offset points", fontsize=7, color=INK_2)
            mark_best(ax, best, "best")
            ax.set_title(f"{title}: {label}")
            if row == 1:
                ax.set_xlabel("optimizer step (effective batch 32 pairs)")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle("From-scratch DPO from the SFT checkpoint (β = 0.1, LR 1e-6 fixed, 15k pairs, 1 epoch)", y=1.0)
    fig.tight_layout()
    save(fig, "dpo_training.png")


def lora_figure():
    runs = [("lora_dpo", "3 epochs, β 0.1", BLUE), ("lora_dpo_1ep", "1 epoch, β 0.1", ORANGE),
            ("lora_dpo_beta05", "3 epochs, β 0.5", AQUA)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for run, label, color in runs:
        beta = read_json(ROOT / f"outputs/{run}/config.json")["beta"]
        rows = read_jsonl(ROOT / f"outputs/{run}/metrics.jsonl")
        evals = [({**r, "step": 0} if r.get("type") == "initial_eval" else r) for r in rows if "eval_loss" in r]
        evals = sorted({r["step"]: r for r in evals}.values(), key=lambda r: r["step"])
        steps = [r["step"] for r in evals]
        series = [
            [r["eval_loss"] for r in evals],
            [r["eval_rewards/accuracies"] for r in evals],
            [r["eval_rewards/margins"] / beta for r in evals],  # log-prob units: comparable across beta
        ]
        for index, (ax, values) in enumerate(zip(axes, series)):
            ax.plot(steps, values, color=color, marker="o", markersize=4, label=label)
            if index != 1:  # accuracy lines coincide at 1.0, so the legend identifies them there
                ax.annotate(label, (steps[-1], values[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7,
                            color=INK_2, va="center")
    for ax, title in zip(axes, ["held-out DPO loss", "held-out reward accuracy", "held-out margin / β (log-prob units)"]):
        ax.set_title(title)
        ax.set_xlabel("optimizer step (effective batch 8)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].annotate("all three runs: 1.0 from step 10 on", (0.6, 0.4), xycoords="axes fraction", ha="center",
                     fontsize=8, color=INK_2)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("LoRA-DPO persona runs on SmolLM2-360M-Instruct (TRL DPOTrainer, 200 train / 20 held-out pairs)", y=1.03)
    fig.tight_layout()
    save(fig, "lora_training.png")


def pilot_figure():
    summary = read_json(ROOT / "outputs/pilot/summary.json")
    settings = [("greedy", "greedy"), ("rp1.1", "rep. penalty\n1.1"), ("rp1.2", "rep. penalty\n1.2"), ("ngram4", "no-repeat\n4-gram")]
    models = [("sft", "SFT", BLUE), ("dpo_uf", "DPO-UF", ORANGE), ("dpo_hh", "DPO-HH", AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    width = 0.26
    for offset, (model, label, color) in enumerate(models):
        xs = [i + (offset - 1) * (width + 0.02) for i in range(len(settings))]
        for ax, key, scale in [(axes[0], "truncated_rate", 100), (axes[1], "mean_distinct_3", 1)]:
            values = [summary[f"{model}__{s}"][key] * scale for s, _ in settings]
            ax.bar(xs, values, width=width, color=color, label=label)
            if key == "truncated_rate":
                for x, value in zip(xs, values):
                    if value == 0:
                        ax.annotate("0", (x, 0), xytext=(0, 2), textcoords="offset points", ha="center",
                                    fontsize=7, color=INK_2)
    axes[0].set(ylabel="% hitting max_new_tokens (512)", title="Responses that never stop (lower is better)")
    axes[1].set(ylabel="distinct word-trigram ratio", title="Lexical repetition (higher = less repetition)", ylim=(0.5, 1.0))
    for ax in axes:
        ax.set_xticks(range(len(settings)), [label for _, label in settings])
    axes[0].legend(loc="upper right")
    fig.suptitle("Decoding pilot: 20 held-out prompts (not in any evaluation set)", y=1.03)
    fig.tight_layout()
    save(fig, "decoding_pilot.png")


def stacked_rows(ax, rows):
    """Horizontal 100% bars: (label, wins, ties, losses)."""

    for y, (label, wins, ties, losses) in enumerate(rows):
        total = wins + ties + losses
        left = 0.0
        for count, color in [(wins, WIN), (ties, TIE), (losses, LOSS)]:
            share = 100 * count / total
            ax.barh(y, share, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2)
            if share >= 7:
                ax.text(left + share / 2, y, str(count), ha="center", va="center", fontsize=8,
                        color="white" if color != TIE else INK)
            left += share
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.axvline(50, color=INK_2, linewidth=0.8, linestyle=":", zorder=0)
    ax.grid(axis="y", visible=False)


def judge_dpo_figure():
    rows = []
    for track, label in [("dpo_uf_vs_sft_alpaca", "DPO-UF vs SFT · AlpacaEval"), ("dpo_hh_vs_sft_hh", "DPO-HH vs SFT · HH set")]:
        for tag, tag_label in [("rp1.1", "rep. penalty 1.1"), ("greedy", "greedy")]:
            c = read_json(ROOT / f"outputs/judged/{tag}/{track}_judged_summary.json")["criteria"]["overall"]
            rate = 100 * c["win_rate_excluding_ties"]
            rows.append((f"{label}\n{tag_label} (win {rate:.1f}% excl. ties)", c["wins"], c["ties"], c["losses"]))
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    stacked_rows(ax, rows)
    ax.set_xlabel("% of 300 pairs")
    ax.set_title("Pairwise judge (google/gemini-3.6-flash): DPO model vs SFT, blind and position-balanced")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (WIN, TIE, LOSS)]
    ax.legend(handles, ["DPO wins", "tie", "SFT wins"], loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    save(fig, "judge_dpo_vs_sft.png")


def judge_categories_figure():
    import collections

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    categories = None
    for tag, color, label, shift in [("rp1.1", BLUE, "rep. penalty 1.1", -0.12), ("greedy", ORANGE, "greedy", 0.12)]:
        counts = collections.defaultdict(collections.Counter)
        for r in read_jsonl(ROOT / f"outputs/judged/{tag}/dpo_hh_vs_sft_hh_judged.jsonl"):
            counts[r["category"]][r["outcomes"]["overall"]] += 1
        categories = categories or sorted(counts)
        rates = [100 * counts[c]["win"] / max(1, counts[c]["win"] + counts[c]["loss"]) for c in categories]
        ax.scatter(rates, [i + shift for i in range(len(categories))], s=46, color=color, label=label, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(50, color=INK_2, linewidth=0.8, linestyle=":")
    ax.set_yticks(range(len(categories)), [c.replace("_", " ") for c in categories])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("DPO-HH win rate vs SFT, excluding ties (%)  ·  ~30 pairs per category")
    ax.set_title("DPO-HH vs SFT by HH-set category")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    save(fig, "judge_hh_categories.png")


def judge_persona_figure():
    runs = [("lora_dpo", "3 epochs, β 0.1"), ("lora_dpo_1ep", "1 epoch, β 0.1"), ("lora_dpo_beta05", "3 epochs, β 0.5")]
    criteria = ["style", "helpfulness", "safety"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8), sharey=True)
    for ax, criterion in zip(axes, criteria):
        rows = []
        for run, label in runs:
            c = read_json(ROOT / f"outputs/judged/rp1.1/{run}_vs_smollm_base_persona_judged_summary.json")["criteria"][criterion]
            rows.append((label, c["wins"], c["ties"], c["losses"]))
        stacked_rows(ax, rows)
        ax.set_title(criterion)
        ax.set_xlabel("% of 24 prompts")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (WIN, TIE, LOSS)]
    fig.legend(handles, ["LoRA-DPO wins", "tie", "base SmolLM2 wins"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Persona LoRA-DPO vs base SmolLM2-360M-Instruct (judge per criterion, rep. penalty 1.1)", y=1.04)
    fig.tight_layout()
    save(fig, "judge_persona.png")


if __name__ == "__main__":
    sft_figure()
    dpo_figure()
    lora_figure()
    pilot_figure()
    judge_dpo_figure()
    judge_categories_figure()
    judge_persona_figure()
