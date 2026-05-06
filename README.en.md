# LLM Finetune Study

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-yellow.svg)](requirements.txt)
[![Platform: macOS | Linux](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)](docs/00-local-first-principles.md)
[![Lessons: 10](https://img.shields.io/badge/lessons-10-brightgreen.svg)](lessons/README.md)

> Hands-on local LLM fine-tuning course (SFT / LoRA / PEFT / DPO / QLoRA) with a browser studio that visualizes loss, tensors and adapter diffs.
> [中文](README.md) · [Discussions](https://github.com/harrisliangsu/llm-finetune-study/discussions) · [Issues](https://github.com/harrisliangsu/llm-finetune-study/issues)

This repo isn't aiming to fine-tune a 7B/13B from day one. It breaks fine-tuning
into pieces you can master on a single laptop:

- Dataset construction, cleaning, splitting, tokenization
- The Hugging Face `datasets` / `transformers` training loop
- Classification fine-tuning, generative fine-tuning, instruction-tuning (SFT)
- LoRA / QLoRA / PEFT — concepts and practical paths
- Chinese instruction data, evaluation, checkpoint and adapter management
- A bridge from local small experiments to DeepSpeed, RLHF, multimodal

## Who this is for

- You want to learn LLM fine-tuning systematically, but only have a laptop
- You'd rather start with small models / small datasets and **really understand the loop** before scaling up
- You want to read open-source Chinese LLM fine-tuning repos and understand their training scripts, data formats, and code structure
- You want a solid fine-tuning baseline before moving on to Agents / RAG / applications

## Who this is NOT for

- Not a guide to large-scale pretraining
- Not for full fine-tuning a 7B/13B locally
- Not chasing leaderboards or hyperparameter cookbooks

## 30-second start

```bash
./bootstrap.sh install      # creates .venv, installs everything (~3GB)
./bootstrap.sh studio       # opens http://127.0.0.1:8765/visualizer/
```

Or, if you only want to look at dataset / tokenizer pieces without GPU deps:

```bash
./bootstrap.sh core         # ~100MB, datasets + tokenizers only
```

`Makefile` shortcuts: `make studio`, `make test`, `make lesson01`, ..., `make lesson10`.

## Learning roadmap

| Stage | Goal | Material |
|---|---|---|
| 0. Local-first principles | What your laptop can / can't run | [docs/00-local-first-principles.md](docs/00-local-first-principles.md) |
| 1. Fine-tuning map | Distinguish full fine-tune / SFT / LoRA / QLoRA / DPO | [docs/01-finetuning-map.md](docs/01-finetuning-map.md) |
| 2. Data & tokenizer | Data formats, prompt assembly, label construction | [docs/02-data-tokenization.md](docs/02-data-tokenization.md) |
| 3. Trainer loop | Run a full classification / generation fine-tune | [docs/03-transformers-trainer.md](docs/03-transformers-trainer.md) |
| 4. SFT + LoRA | Step into instruction-tuning and PEFT | [docs/04-sft-lora.md](docs/04-sft-lora.md) |
| 5. Reading reference repos | Open-source Chinese LLM fine-tuning projects | [docs/05-reference-repos.md](docs/05-reference-repos.md) |
| 6. Evaluation & debugging | Loss curves, samples, metrics, overfitting, data issues | [docs/06-evaluation-debugging.md](docs/06-evaluation-debugging.md) |
| 7. Training methods guide | Choosing between SFT, PEFT/LoRA, DPO, RLHF, distillation | [docs/07-training-methods-guide.md](docs/07-training-methods-guide.md) |
| Tool. Training studio | Web UI to configure runs, validate JSONL, inspect adapter / report / trace | [docs/08-training-studio.md](docs/08-training-studio.md) |

## LLM Study Studio

The `visualizer/` folder ships a local web studio that puts "configure a training
run" and "read course traces" in the same window.

```bash
./bootstrap.sh studio
# opens http://127.0.0.1:8765/visualizer/
```

Two pages, intentionally separated:

| Page | Purpose | Artifacts |
|---|---|---|
| Train | Default landing page. Pick method, model, JSONL data, basic + advanced params, launch a self-contained Studio run. | `visualizer/runtime/studio-runs/<run-id>/` |
| Lessons | Watch traces from lessons 01–10, run lesson scripts, inspect data flow / tensors / loss / adapter / checkpoint / Chat Lab. | `visualizer/traces/live.json`, `visualizer/traces/<lesson-id>.json` |

Studio runs and lesson runs are isolated by design:

- Studio entry: `visualizer/studio/run.py`; engines: `training/engines/`; helpers: `training/common/`.
- Studio never calls `lessons/` scripts and never overwrites lesson reports or archived traces.
- Each Studio run is self-contained: data copy, trace, report, metrics, generations, adapter or QLoRA plan.
- The Train page's Chat compare always shows base / adapter / reloaded-adapter side by side, so you can verify the adapter path actually reloads.

Currently supported Studio methods:

| Method | Data format | Notes |
|---|---|---|
| SFT + LoRA | `instruction/input/output` JSONL | Strict-JSON ticket routing SFT, saves a LoRA adapter. |
| PEFT LoRA | `instruction/input/output` JSONL | Real Hugging Face causal LM + PEFT LoRA. |
| DPO Preference | `instruction/input/chosen/rejected` JSONL | Preference objective without TRL — policy / reference / adapter. |
| QLoRA Plan | No JSONL | Generates a local / CUDA / VRAM / quantization plan. Doesn't fake bitsandbytes 4-bit on Mac/MPS. |

See [docs/08-training-studio.md](docs/08-training-studio.md) for full usage,
schema, layout, and troubleshooting; [visualizer/README.md](visualizer/README.md)
for the page / API contract.

## The 10 lessons

| # | Topic | Path | Focus |
|---|---|---|---|
| 01 | Datasets pipeline | [01-datasets](lessons/01-datasets) | JSONL → Dataset → split → filter → map |
| 02 | AutoTokenizer | [02-tokenizer](lessons/02-tokenizer) | text → input_ids / attention_mask / labels |
| 03 | Batch / Collator | [03-batching](lessons/03-batching) | list[dict] → dict[tensor] |
| 04 | Trainer loop | [04-trainer](lessons/04-trainer) | model + dataset + loss + optimizer |
| 05 | LoRA Adapter | [05-lora](lessons/05-lora) | frozen base + trainable adapter |
| 06 | PEFT LoRA | [06-peft-lora](lessons/06-peft-lora) | real HF model + PEFT adapter |
| 07 | SFT Baseline | [07-sft-baseline](lessons/07-sft-baseline) | ticket text → strict JSON |
| 08 | DPO | [08-dpo-preference](lessons/08-dpo-preference) | prompt + chosen/rejected → preference margin |
| 09 | Reward / RLHF | [09-rlhf-reward](lessons/09-rlhf-reward) | reward model, reference model, KL, PPO signals |
| 10 | QLoRA / Engineering | [10-qlora-engineering](lessons/10-qlora-engineering) | quantization, memory budget, CUDA / MPS boundary |

Run them all in order:

```bash
make test              # smoke (compileall + run_all --help)
.venv/bin/python lessons/run_all.py
.venv/bin/python lessons/run_all.py --quick   # quick sanity pass
```

Each lesson directory contains: `README.md`, `run.py`, `report.md`, `index.html`, `outputs/`.

## Model defaults

When a lesson needs a model, we prefer **real Hugging Face checkpoints** over
random-initialized ones:

- Lesson 04: `sshleifer/tiny-gpt2` — fast Trainer-loop study.
- Lesson 06: `--model-name auto` resolves to `Qwen/Qwen2.5-0.5B-Instruct` on a 32GB MPS box; real PEFT LoRA.
- Lesson 07: same `auto` rule; Chinese strict-JSON SFT baseline.
- Lesson 08: `auto`, real HF causal LM for DPO.
- Lesson 09: lightweight local reward / KL / PPO signal — full RLHF would not fit a Mac.
- Lesson 10: a hardware + training-engineering plan; clarifies QLoRA's CUDA + bitsandbytes path vs Mac / MPS.

Lesson 05 keeps a hand-written LoRA on purpose, to teach the role of the
low-rank `A/B` matrices. It's not a real-model training lesson.

## Reference repos

Picked from personal stars, biased toward learning:

- [huggingface/datasets](https://github.com/huggingface/datasets)
- [huggingface/transformers](https://github.com/huggingface/transformers)
- [Lordog/dive-into-llms](https://github.com/Lordog/dive-into-llms)
- [ymcui/Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE)
- [LlamaChinese/Llama-Chinese](https://github.com/LlamaChinese/Llama-Chinese)
- [LAION-AI/Open-Assistant](https://github.com/LAION-AI/Open-Assistant)
- [deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed)

Reading guide: [docs/05-reference-repos.md](docs/05-reference-repos.md).

## The local learning principle

Learning fine-tuning on a laptop, the goal isn't **big** — it's **complete**:

- You can explain why each field reaches the model
- You can explain which tokens contribute to loss
- You can read a training curve and an eval result
- You can save / load / resume a checkpoint
- You can reproduce a minimal SFT + LoRA flow

Truly understanding a small model beats blindly launching a big model that
won't run.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and PRs welcome —
ideas / questions go to [Discussions](https://github.com/harrisliangsu/llm-finetune-study/discussions).
