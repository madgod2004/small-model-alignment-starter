"""Build outputs/report_tables.md: every number the report needs, computed from committed artifacts.

    python scripts/make_tables.py

Sections: data, experimental configuration (values actually used), training results, decoding pilot,
generation statistics, DPO-vs-SFT judge results (with 95% CIs and sign tests), per-category HH
results, persona results, and judge diagnostics.
"""

from __future__ import annotations

import collections
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alignment.text_stats import summarize  # noqa: E402
from alignment.utils import read_json, read_jsonl  # noqa: E402

TAGS = [("rp1.1", "rep. penalty 1.1"), ("greedy", "greedy")]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def sign_test(k: int, n: int) -> float:
    """Two-sided exact binomial test against 0.5."""

    pmf = [math.comb(n, i) / 2**n for i in range(n + 1)]
    return min(1.0, sum(p for p in pmf if p <= pmf[k] * (1 + 1e-9)))


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def section_data() -> str:
    rows = []
    for name, label in [("sft", "SFT"), ("hh", "HH-RLHF DPO"), ("uf", "UltraFeedback DPO")]:
        m = read_json(ROOT / f"data/manifests/{name}.json")
        tr, va = m["splits"]["train"], m["splits"]["val"]
        dropped = {k: v for k, v in tr["stats"].items() if k not in ("visited", "selected")}
        rows.append([label, m["source_dataset"], f"{tr['source_split']} / {va['source_split']}",
                     tr["stats"]["selected"], va["stats"]["selected"], m["seed"],
                     ", ".join(f"{k}={v}" for k, v in dropped.items()), (m["source_revision"] or "")[:10]])
    lora = read_json(ROOT / "outputs/lora_dpo/summary.json")
    rows.append(["LoRA persona DPO", "data/persona_lora/ramsay_style_preferences.json", "seeded split",
                 lora["train_pairs"], lora["eval_pairs"], 42, "-", "-"])
    out = table(["track", "source", "train / val split", "train", "val", "seed", "train-split drops", "revision"], rows)
    evals = [["AlpacaEval (UF track)", "eval_sets/alpaca_eval_300_seed42.json", 300],
             ["HH helpful/harmless (HH track)", "eval_sets/hh_rlhf_eval_300_seed42.json", 300],
             ["Persona style prompts (LoRA track)", "eval_sets/persona_style_eval.json", 24],
             ["Decoding pilot (held out)", "eval_sets/decoding_pilot_20.json", 20]]
    return "## Data\n\n" + out + "\n\n" + table(["evaluation set", "file", "prompts"], evals)


def section_config() -> str:
    sft = read_json(ROOT / "outputs/sft/config.json")
    hh, uf = read_json(ROOT / "outputs/dpo_hh/config.json"), read_json(ROOT / "outputs/dpo_uf/config.json")
    lora = {r: read_json(ROOT / f"outputs/{r}/config.json") for r in ["lora_dpo", "lora_dpo_1ep", "lora_dpo_beta05"]}
    rows = [
        ["SFT", "full fine-tuning (fp32 weights)", "AdamW", sft["learning_rate"], f"{sft['lr_scheduler']}, warmup {sft['warmup_ratio']}",
         sft["weight_decay"], sft["max_grad_norm"], f"{sft['per_device_train_batch_size']} x {sft['gradient_accumulation_steps']} = {sft['effective_batch_size']}",
         "fp16 autocast + GradScaler", sft["max_length"], f"1 epoch ({sft['total_optimizer_steps']} steps)", "-", "-", sft["seed"]],
    ]
    for label, c in [("HH-RLHF DPO", hh), ("UltraFeedback DPO", uf)]:
        rows.append([label, "full fine-tuning (fp32 policy)", c["optimizer"].replace("adamw", "AdamW"), c["learning_rate"],
                     f"{c['lr_scheduler']} (no warmup)", c["weight_decay"], c["max_grad_norm"],
                     f"{c['per_device_train_batch_size']} x {c['gradient_accumulation_steps']} = {c['effective_batch_size']}",
                     f"fp16 autocast; reference {c['reference_dtype']} on cuda:1", f"{c['max_length']} (prompt {c['max_prompt_length']})",
                     f"1 epoch ({c['total_optimizer_steps']} steps)", c["beta"], "SFT best (step 938)", c["seed"]])
    for run, c in lora.items():
        rows.append([f"LoRA-DPO ({run})",
                     f"LoRA r={c['lora_r']}, alpha={c['lora_alpha']}, dropout={c['lora_dropout']}, {'/'.join(c['lora_target_modules'])}",
                     "AdamW (TRL default)", c["learning_rate"], f"{c['lr_scheduler_type']}, warmup {c['warmup_ratio']}",
                     c["weight_decay"], c["max_grad_norm"],
                     f"{c['per_device_train_batch_size']} x {c['gradient_accumulation_steps']} = {c['per_device_train_batch_size'] * c['gradient_accumulation_steps']}",
                     "fp16 autocast", c["max_length"], f"{c['num_train_epochs']} epoch(s)", c["beta"],
                     "base SmolLM2 (adapter disabled)", c["seed"]])
    header = ["run", "trainable params", "optimizer", "peak LR", "schedule", "weight decay", "grad clip",
              "batch (per-device x accum = effective)", "precision", "max length", "epochs / steps", "beta", "reference", "seed"]
    decoding = ("Generation (all models): greedy (do_sample=false, temperature/top-p unused), max_new_tokens 512, fp16, batch 16. "
                "Main setting: repetition_penalty 1.1 (`configs/generation_rp1.1.yaml`); sensitivity check: repetition_penalty 1.0 "
                "(`configs/generation.yaml`). No length penalty, no n-gram blocking. Prompt template: Qwen ChatML with system "
                "message \"You are a helpful assistant.\" (`alignment/prompting.py`); SmolLM2 uses its own tokenizer chat template.")
    return "## Experimental configuration (values actually used)\n\n" + table(header, rows) + "\n\n" + decoding


def section_training() -> str:
    sft = read_json(ROOT / "outputs/sft/summary.json")
    sft_evals = [r for r in read_jsonl(ROOT / "outputs/sft/metrics.jsonl") if r["type"] == "eval"]
    rows = [["SFT", sft["steps_completed"], f"val NLL {sft_evals[0]['val_loss']:.4f} (base) -> {sft['final_val_loss']:.4f}",
             f"step {sft['best_step']} (val NLL {sft['best_val_loss']:.4f})", f"{sft['train_seconds'] / 60:.0f} min",
             f"{sft['max_memory_gb']:.1f} GB", sft["skipped_steps"], f"{sft['train_examples_truncated']}/{sft['train_examples']} examples"]]
    for track, label in [("hh", "HH-RLHF DPO"), ("uf", "UltraFeedback DPO")]:
        s = read_json(ROOT / f"outputs/dpo_{track}/summary.json")
        ev = [r for r in read_jsonl(ROOT / f"outputs/dpo_{track}/metrics.jsonl") if r["type"] == "eval"]
        best = next(r for r in ev if r["step"] == s["best_step"])
        rows.append([label, s["steps_completed"],
                     f"val loss {ev[0]['val_loss']:.4f} -> {s['final_val_metrics']['loss']:.4f}; acc {ev[1]['val_reward_accuracy']:.3f} (step 50) -> {s['final_val_metrics']['reward_accuracy']:.3f}; margin {s['final_val_metrics']['reward_margin']:.3f}",
                     f"step {s['best_step']} (acc {best['val_reward_accuracy']:.3f}, margin {best['val_reward_margin']:.3f}, chosen reward {best['val_chosen_rewards']:+.3f}, rejected {best['val_rejected_rewards']:+.3f})",
                     f"{s['train_seconds'] / 60:.0f} min", f"{s['max_memory_gb_policy']:.1f} / {s['max_memory_gb_reference']:.1f} GB", s["skipped_steps"],
                     f"{s['train_pairs_truncated']}/{s['train_pairs']} pairs"])
    for run in ["lora_dpo", "lora_dpo_1ep", "lora_dpo_beta05"]:
        s = read_json(ROOT / f"outputs/{run}/summary.json")
        beta = read_json(ROOT / f"outputs/{run}/config.json")["beta"]
        e = s["final_eval"]
        rows.append([f"LoRA-DPO ({run})", s["optimizer_steps"],
                     f"held-out loss {s['initial_eval']['eval_loss']:.4f} -> {e['eval_loss']:.4f}; acc {e['eval_rewards/accuracies']:.2f}; margin {e['eval_rewards/margins']:.3f} (= {e['eval_rewards/margins'] / beta:.1f} log-prob)",
                     "final (no selection)", f"{s['train_seconds']:.0f} s", "-", "-",
                     f"{s['trainable_parameters']:,} trainable ({s['trainable_percent']:.2f}%)"])
    return "## Training results\n\n" + table(["run", "steps", "trajectory", "selected checkpoint", "time", "peak memory", "fp16 skipped steps", "truncated at max length / notes"], rows)


def section_pilot() -> str:
    s = read_json(ROOT / "outputs/pilot/summary.json")
    rows = []
    for setting, label in [("greedy", "greedy"), ("rp1.1", "rep. penalty 1.1"), ("rp1.2", "rep. penalty 1.2"), ("ngram4", "no-repeat 4-gram")]:
        rows.append([label] + [f"{pct(s[f'{m}__{setting}']['truncated_rate'])}% / {s[f'{m}__{setting}']['mean_distinct_3']:.3f} / {s[f'{m}__{setting}']['mean_tokens']:.0f}"
                               for m in ["sft", "dpo_uf", "dpo_hh"]])
    return ("## Decoding pilot (20 held-out prompts)\n\nCells: % hitting max_new_tokens / distinct-3 / mean tokens.\n\n"
            + table(["setting", "SFT", "DPO-UF", "DPO-HH"], rows))


def section_generation_stats() -> str:
    rows = []
    for name in ["base_alpaca_eval", "sft_alpaca_eval", "dpo_uf_alpaca_eval", "base_hh_eval", "sft_hh_eval", "dpo_hh_hh_eval",
                 "smollm_base_persona_eval", "lora_dpo_persona_eval", "lora_dpo_1ep_persona_eval", "lora_dpo_beta05_persona_eval"]:
        g = summarize(read_json(ROOT / f"outputs/generations/{name}.json"))
        r = summarize(read_json(ROOT / f"outputs/generations_rp1.1/{name}.json"))
        rows.append([name, g["n"], f"{pct(g['truncated_rate'])} / {pct(r['truncated_rate'])}", f"{g['mean_tokens']:.0f} / {r['mean_tokens']:.0f}",
                     f"{g['mean_repeated_line_rate']:.3f} / {r['mean_repeated_line_rate']:.3f}", f"{g['mean_distinct_3']:.3f} / {r['mean_distinct_3']:.3f}"])
    return ("## Generation statistics (greedy / rep. penalty 1.1)\n\n"
            + table(["generation file", "n", "% hitting max tokens", "mean tokens", "repeated-line rate", "distinct-3"], rows))


def judge_rows(files: list[tuple[str, str]], criterion: str = "overall") -> list[list]:
    rows = []
    for name, label in files:
        for tag, tag_label in TAGS:
            s = read_json(ROOT / f"outputs/judged/{tag}/{name}_judged_summary.json")
            c = s["criteria"][criterion]
            lo, hi = wilson(c["wins"], c["wins"] + c["losses"])
            rows.append([label, tag_label, c["n"], c["wins"], c["ties"], c["losses"], c["unknowns"], c["failures"],
                         pct(c["win_rate_excluding_ties"]), f"[{lo:.1f}, {hi:.1f}]", pct(c["win_rate_ties_as_half"]),
                         f"{sign_test(c['wins'], c['wins'] + c['losses']):.3f}", s["retried_pairs"]])
    return rows


JUDGE_HEADER = ["comparison", "decoding", "n", "wins", "ties", "losses", "unknown", "failed",
                "win % excl. ties", "95% CI (Wilson)", "win % ties = 1/2", "sign-test p", "retried pairs"]


def section_judge() -> str:
    rows = judge_rows([("dpo_uf_vs_sft_alpaca", "DPO-UF vs SFT (AlpacaEval 300)"), ("dpo_hh_vs_sft_hh", "DPO-HH vs SFT (HH set 300)")])
    note = ("Judge: google/gemini-3.6-flash via kaggle_benchmarks, temperature 0; 5 pairs per request, >= 5 s between requests, "
            "up to 3 attempts per batch; A/B order exactly balanced (150/150) with identities hidden. 'Wins' are wins of the DPO model.")
    return "## DPO vs SFT (pairwise judge)\n\n" + note + "\n\n" + table(JUDGE_HEADER, rows)


def section_categories() -> str:
    out = ["## DPO-HH vs SFT by HH category (W / T / L)\n"]
    counts = {}
    for tag, _ in TAGS:
        c = collections.defaultdict(collections.Counter)
        for r in read_jsonl(ROOT / f"outputs/judged/{tag}/dpo_hh_vs_sft_hh_judged.jsonl"):
            c[r["category"]][r["outcomes"]["overall"]] += 1
        counts[tag] = c
    rows = []
    for category in sorted(counts["rp1.1"]):
        cells = []
        for tag, _ in TAGS:
            v = counts[tag][category]
            rate = v["win"] / max(1, v["win"] + v["loss"])
            cells.append(f"{v['win']} / {v['tie']} / {v['loss']} ({100 * rate:.0f}%)")
        rows.append([category] + cells)
    out.append(table(["category (~30 each)", "rep. penalty 1.1", "greedy"], rows))
    return "\n".join(out)


def section_persona() -> str:
    files = [("lora_dpo_vs_smollm_base_persona", "LoRA 3 epochs, beta 0.1"),
             ("lora_dpo_1ep_vs_smollm_base_persona", "LoRA 1 epoch, beta 0.1"),
             ("lora_dpo_beta05_vs_smollm_base_persona", "LoRA 3 epochs, beta 0.5")]
    rows = []
    for name, label in files:
        for tag, tag_label in TAGS:
            s = read_json(ROOT / f"outputs/judged/{tag}/{name}_judged_summary.json")["criteria"]
            rows.append([label, tag_label] + [f"{s[c]['wins']} / {s[c]['ties']} / {s[c]['losses']}" for c in ["style", "helpfulness", "safety"]])
    return ("## Persona LoRA-DPO vs base SmolLM2 (24 prompts; W / T / L for the LoRA model)\n\n"
            + table(["run", "decoding", "style", "helpfulness", "safety"], rows))


def section_diagnostics() -> str:
    rows = []
    for tag, tag_label in TAGS:
        for path in sorted((ROOT / f"outputs/judged/{tag}").glob("*_summary.json")):
            s = read_json(path)
            rows.append([path.name.replace("_judged_summary.json", ""), tag_label,
                         ", ".join(f"{c}: {pct(v)}%" for c, v in s["judge_picked_A_rate"].items())])
    agreement = []
    for name in ["dpo_uf_vs_sft_alpaca", "dpo_hh_vs_sft_hh"]:
        g = {r["pair_id"]: r["outcomes"]["overall"] for r in read_jsonl(ROOT / f"outputs/judged/greedy/{name}_judged.jsonl")}
        r_ = {r["pair_id"]: r["outcomes"]["overall"] for r in read_jsonl(ROOT / f"outputs/judged/rp1.1/{name}_judged.jsonl")}
        agreement.append([name, f"{sum(g[k] == r_[k] for k in g)}/{len(g)}"])
    return ("## Judge diagnostics\n\nShare of decided verdicts that picked 'Response A' (positions are balanced 50/50, so "
            "this measures position bias, which cancels in the win rates):\n\n" + table(["file", "decoding", "picked A"], rows)
            + "\n\nSame outcome for the same prompt under greedy and rp1.1 generations:\n\n" + table(["comparison", "agreement"], agreement))


def main() -> None:
    sections = [section_data(), section_config(), section_training(), section_pilot(), section_generation_stats(),
                section_judge(), section_categories(), section_persona(), section_diagnostics()]
    text = "# Report tables (generated by scripts/make_tables.py)\n\n" + "\n\n".join(sections) + "\n"
    out = ROOT / "outputs" / "report_tables.md"
    out.write_text(text, encoding="utf-8")
    print("wrote", out.relative_to(ROOT), f"({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
