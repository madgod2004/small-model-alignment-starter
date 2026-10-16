# Small-Model Alignment: SFT + DPO on Qwen2.5-0.5B

Post-training a small pretrained model in two stages, following `docs/assignment_manual.pdf`:
supervised fine-tuning (SFT) on 30k instruction–response pairs, then two **from-scratch DPO** runs
(HH-RLHF and UltraFeedback) from the SFT checkpoint, plus a **TRL `DPOTrainer` + LoRA** persona run on
SmolLM2-360M-Instruct. Models are compared with blind, position-balanced pairwise judgments from a
Kaggle model-proxy judge (`google/gemini-3.6-flash`).

## Where the implementation lives

| What | File |
|---|---|
| Prompt template (Qwen ChatML, fixed system message) | `alignment/prompting.py` |
| Tokenisation, EOS-before-truncation, padding, `-100` response-only labels | `alignment/batching.py` |
| Response log-probs, SFT loss (token-mean), DPO loss + implicit-reward metrics | `alignment/losses.py` |
| **Grading interface** (thin wrappers over the above) | `submission_adapter.py` |
| SFT training loop (no HF Trainer): grad accumulation, fp16 autocast + GradScaler, clipping, val loss, best/final checkpoints | `alignment/sft.py`, `scripts/train_sft.py` |
| DPO training loop (no TRL): policy on cuda:0, frozen reference on cuda:1, best checkpoint by val reward accuracy | `alignment/dpo.py`, `scripts/train_dpo.py` |
| Optimizer, schedule, AMP, checkpoint helpers | `alignment/training.py` |
| LoRA persona DPO with TRL `DPOTrainer` + PEFT | `scripts/train_lora_dpo.py` |
| Dataset parsing / seeded subsampling | `alignment/data.py`, `scripts/prepare_data.py` |
| Generation (greedy, left-padded batches, resumable) | `alignment/generation.py`, `scripts/generate.py`, `scripts/run_generation_jobs.py` |
| Blind pairwise files (balanced A/B, identities hidden) | `alignment/pairs.py`, `scripts/build_pairs.py` |
| Pairwise judge (batches of 5, ≥5 s sleep, 3 attempts, failure log, resume) | `alignment/judging.py`, `scripts/judge_pairwise_kaggle.py` |
| Figures and report tables from committed artifacts | `scripts/make_figures.py`, `scripts/make_tables.py` |

`compute_sft_loss` uses the **mean over all unmasked response tokens in the batch**.

## Pipeline (Kaggle, GPU T4 x2)

Run the notebooks in order; each later notebook attaches the saved output of the earlier ones.

| Notebook | Task | Inputs | Main outputs |
|---|---|---|---|
| `00_kaggle_setup` | 1 — clone, install (no torch reinstall), GPU + version check | – | `outputs/environment.md` |
| `01_baseline_generation` | 2 — base-model generations | – | `outputs/generations/base_*` |
| `02_train_sft` | 3 — data subsets + SFT (1 GPU, batch 2 x 16) | – | `outputs/sft/`, `data/manifests/` |
| `03_train_dpo` | 4 — HH-RLHF and UltraFeedback DPO | 02 | `outputs/dpo_hh/`, `outputs/dpo_uf/` |
| `04_lora_persona_dpo`, `04b_lora_persona_variants` | 5 — TRL + LoRA persona DPO (3 runs) | – | `outputs/lora_dpo*/` |
| `05a_decoding_pilot` | choose anti-repetition decoding on 20 held-out prompts | 02, 03 | `outputs/pilot/` |
| `05_generate_eval`, `05b_generate_eval_rp1.1` | 6 — eval generations + blind pairs (greedy, rep. penalty 1.1) | 02, 03, 04, 04b | `outputs/generations*/`, `outputs/eval*/` |
| `06_judge_pairwise` (Kaggle **benchmark** notebook) | 7 — pairwise judging | – | `outputs/judged/` |

Locally (no GPU):

```bash
python -m pip install -r requirements.txt && python -m pip install -e . --no-deps
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
python scripts/make_figures.py   # outputs/figures/*.png
python scripts/make_tables.py    # outputs/report_tables.md
```

## Results at a glance

Pairwise judge, DPO model vs SFT on the same prompts (wins / ties / losses for the DPO model):

| Comparison | Decoding | W / T / L | Win % excl. ties (95% CI) |
|---|---|---|---|
| DPO-UF vs SFT, AlpacaEval 300 (seed 42) | rep. penalty 1.1 | 105 / 35 / 160 | 39.6 (33.9–45.6) |
| DPO-UF vs SFT | greedy | 118 / 36 / 146 | 44.7 (38.8–50.7) |
| DPO-HH vs SFT, HH set 300 | rep. penalty 1.1 | 111 / 62 / 127 | 46.6 (40.4–53.0) |
| DPO-HH vs SFT | greedy | 125 / 39 / 136 | 47.9 (41.9–53.9) |

Persona (LoRA 3 epochs, β 0.1 vs base SmolLM2, rep. penalty 1.1): style 7 / 17 / 0, helpfulness 10 / 2 / 12, safety 4 / 17 / 3.
Full tables (configurations, training metrics, per-category results, judge diagnostics): `outputs/report_tables.md`.

## Result files

| Artifact | Path |
|---|---|
| Configs actually used | `configs/*.yaml`, `outputs/*/config.json` |
| Training metrics / summaries | `outputs/{sft,dpo_hh,dpo_uf,lora_dpo,lora_dpo_1ep,lora_dpo_beta05}/{metrics.jsonl,summary.json}` |
| Data subsets (sources, revisions, seeds, filters, selected indices) | `data/manifests/*.json` |
| Fixed evaluation sets | `eval_sets/` |
| Generations (+ `.meta.json` with decoding settings) | `outputs/generations/` (greedy), `outputs/generations_rp1.1/` |
| Blind pairwise files | `outputs/eval/`, `outputs/eval_rp1.1/` |
| Judgments + summaries | `outputs/judged/{greedy,rp1.1}/` |
| Decoding pilot | `outputs/pilot/` |
| Figures | `outputs/figures/` |
| Logs | `outputs/logs/` |

Checkpoints (~2 GB each) are not in git; they are kept in the saved Kaggle notebook outputs.

## Deviations from the guide configuration

- **Prompt filter:** training examples whose templated prompt exceeds 256 tokens are dropped (all tracks), so every response keeps ≥ 256 of the 512 tokens; UltraFeedback pairs with tied scores are dropped.
- **DPO reference in fp32** (guide allows fp16) so policy and reference are identical at step 0 (loss ln 2, accuracy 0).
- **Evaluation decoding:** greedy with repetition penalty 1.1 as the main setting (greedy loops in 20–30% of responses in the pilot); plain greedy reported as a sensitivity check.
- **LoRA persona:** two extra variants (1 epoch; β 0.5) to test over-optimisation of the 3-epoch run.

## Environment

Kaggle T4 x2, torch 2.11.0+cu128, transformers 4.57.6, datasets 4.8.5, peft 0.20.0, trl 1.15.0 (`outputs/environment.md`).
