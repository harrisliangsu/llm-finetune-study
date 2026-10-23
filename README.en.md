# LLM Finetune Study

[![Stars](https://img.shields.io/github/stars/harrisliangsu/llm-finetune-study?style=flat-square)](https://github.com/harrisliangsu/llm-finetune-study/stargazers)
[![Forks](https://img.shields.io/github/forks/harrisliangsu/llm-finetune-study?style=flat-square)](https://github.com/harrisliangsu/llm-finetune-study/forks)
[![License: MIT](https://img.shields.io/github/license/harrisliangsu/llm-finetune-study?style=flat-square)](LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/harrisliangsu/llm-finetune-study?style=flat-square)](https://github.com/harrisliangsu/llm-finetune-study/commits/main)
[![Read online](https://img.shields.io/badge/read-GitHub%20Pages-c2410c?style=flat-square)](https://harrisliangsu.github.io/llm-finetune-study/)

> A local, Chinese-first LLM fine-tuning course: ten runnable lessons from data, tokenizers, and Trainer through SFT, LoRA / PEFT, DPO, and QLoRA planning.
> [中文](README.md) · [Read online](https://harrisliangsu.github.io/llm-finetune-study/) · [Discussions](https://github.com/harrisliangsu/llm-finetune-study/discussions) · [Issues](https://github.com/harrisliangsu/llm-finetune-study/issues)

Sponsor: [Afdian](https://afdian.com/a/sharkchiligaga)

This repo is not a plan to full fine-tune 7B/13B on day one. The browser studio shows loss, tensors, and adapters, and it has to run on your machine. GitHub Pages cannot train.

## Contents

- [Read online](#read-online)
- [Who this is for](#who-this-is-for)
- [30-second start](#30-second-start)
- [Learning roadmap](#learning-roadmap)
- [Suggested order](#suggested-order)
- [LLM Study Studio](#llm-study-studio)
- [The 10 lessons](#the-10-lessons)
- [Model defaults](#model-defaults)
- [Reference repos](#reference-repos)
- [Community](#community)

## Read online

Lesson HTML is published from the repository root:

**https://harrisliangsu.github.io/llm-finetune-study/**

The site root is `index.html`, with `.nojekyll` so the existing HTML is served as-is. Source branch is `main`, folder `/ (root)`.

Pages is the lesson reader. Markdown docs stay on GitHub. `visualizer/index.html` calls a local API, so it is not an online studio.

| # | Topic | HTML | Repo |
|---|---|---|---|
| 01 | Datasets pipeline | [Read](https://harrisliangsu.github.io/llm-finetune-study/lessons/01-datasets/index.html) | [01-datasets](lessons/01-datasets) |
| 02 | AutoTokenizer | [Read](https://harrisliangsu.github.io/llm-finetune-study/lessons/02-tokenizer/index.html) | [02-tokenizer](lessons/02-tokenizer) |
| 03 | Batch / Collator | [Read](https://harrisliangsu.github.io/llm-finetune-study/lessons/03-batching/index.html) | [03-batching](lessons/03-batching) |
| 04 | Trainer loop | [Read](https://harrisliangsu.github.io/llm-finetune-study/lessons/04-trainer/index.html) | [04-trainer](lessons/04-trainer) |
| 05 | Hand-written LoRA | [Read](https://harrisliangsu.github.io/llm-finetune-study/lessons/05-lora/index.html) | [05-lora](lessons/05-lora) |
| 06 | PEFT LoRA | [Read](https://harrisliangsu.github.io/llm-finetune-study/lessons/06-peft-lora/index.html) | [06-peft-lora](lessons/06-peft-lora) |
| 07 | SFT baseline | [Read](https://harrisliangsu.github.io/llm-finetune-study/lessons/07-sft-baseline/index.html) | [07-sft-baseline](lessons/07-sft-baseline) |
| 08 | DPO | [Read](https://harrisliangsu.github.io/llm-finetune-study/lessons/08-dpo-preference/index.html) | [08-dpo-preference](lessons/08-dpo-preference) |
| 09 | Reward / RLHF concepts | [Read](https://harrisliangsu.github.io/llm-finetune-study/lessons/09-rlhf-reward/index.html) | [09-rlhf-reward](lessons/09-rlhf-reward) |
| 10 | QLoRA engineering | [Read](https://harrisliangsu.github.io/llm-finetune-study/lessons/10-qlora-engineering/index.html) | [10-qlora-engineering](lessons/10-qlora-engineering) |

Docs to open on GitHub before you train:

- [ROADMAP.md](ROADMAP.md)
- [docs/00-local-first-principles.md](docs/00-local-first-principles.md)
- [docs/01-finetuning-map.md](docs/01-finetuning-map.md)
- [docs/07-training-methods-guide.md](docs/07-training-methods-guide.md)
- [docs/08-training-studio.md](docs/08-training-studio.md)
- [checklists/local-finetuning-checklist.md](checklists/local-finetuning-checklist.md)

Practice order: [examples/README.md](examples/README.md).

## Who this is for

- You want to learn LLM fine-tuning systematically, but only have a laptop
- You'd rather start with small models and small datasets and understand the loop before scaling up
- You want to read open-source Chinese LLM fine-tuning repos and understand their scripts, data formats, and layout
- You want a fine-tuning baseline before Agents, RAG, or applications

This is not a large-scale pretraining guide, and it is not a recipe for full fine-tuning a 7B/13B locally. Leaderboards and hyperparameter cookbooks are not the first goal.

## 30-second start

```bash
./bootstrap.sh install      # creates .venv, installs everything (about 3GB, including torch)
./bootstrap.sh studio       # opens http://127.0.0.1:8765/visualizer/
```

Dataset and tokenizer only, without the training stack:

```bash
./bootstrap.sh core         # about 100MB, datasets + tokenizers
```

`Makefile` shortcuts: `make studio`, `make test`, `make lesson01` … `make lesson10`.

Preview the lesson HTML locally (this is still not the studio):

```bash
python3 -m http.server 8000
# http://127.0.0.1:8000
```

## Learning roadmap

| Stage | Goal | Material |
|---|---|---|
| 0. Local-first principles | What your laptop can and cannot run | [docs/00-local-first-principles.md](docs/00-local-first-principles.md) |
| 1. Fine-tuning map | Distinguish full fine-tune, SFT, LoRA, QLoRA, DPO | [docs/01-finetuning-map.md](docs/01-finetuning-map.md) |
| 2. Data and tokenizer | Data formats, prompt assembly, label construction | [docs/02-data-tokenization.md](docs/02-data-tokenization.md) |
| 3. Trainer loop | Run a classification or generation fine-tune | [docs/03-transformers-trainer.md](docs/03-transformers-trainer.md) |
| 4. SFT + LoRA | Instruction tuning and PEFT | [docs/04-sft-lora.md](docs/04-sft-lora.md) |
| 5. Reading reference repos | Open-source Chinese LLM fine-tuning projects | [docs/05-reference-repos.md](docs/05-reference-repos.md) |
| 6. Evaluation and debugging | Loss, samples, metrics, overfitting, data issues | [docs/06-evaluation-debugging.md](docs/06-evaluation-debugging.md) |
| 7. Training methods guide | Choosing SFT, PEFT/LoRA, DPO, RLHF, distillation | [docs/07-training-methods-guide.md](docs/07-training-methods-guide.md) |
| Tool. Training studio | Configure a run, validate JSONL, inspect adapter / report / trace | [docs/08-training-studio.md](docs/08-training-studio.md) |

## Suggested order

1. Learn the training loop on a small classical ML or NLP task.
2. Fine-tune a text classifier with `datasets` + `transformers`.
3. Fine-tune a small causal LM.
4. Build a small SFT set in `instruction` / `input` / `output`.
5. Compare before and after on a fixed prompt set.
6. Then LoRA / QLoRA, DPO, Chinese instruction data, and adapter save / merge.

Lessons 02–06 already use SFT-shaped data and labels, while each lesson title stays on tokenizer, batch, Trainer, LoRA, or PEFT. Lesson 07 is the standalone SFT baseline: support tickets to strict JSON, fixed eval prompts, and a before/after check of JSON validity, fields, intent, and department. Adapter save / load / reload is covered in lessons 06 and 07. DPO comes after that.

## LLM Study Studio

The `visualizer/` app is a local web studio. It is not part of GitHub Pages.

```bash
./bootstrap.sh studio
# or: .venv/bin/python visualizer/serve.py
# http://127.0.0.1:8765/visualizer/
```

| Page | Purpose | Artifacts |
|---|---|---|
| Train | Default page. Pick method, model, JSONL, basic and advanced params, and launch one self-contained Studio run. | `visualizer/runtime/studio-runs/<run-id>/` |
| Lessons | Watch traces for lessons 01–10, run lesson scripts, inspect data flow, tensors, loss, adapter, checkpoint, and Chat Lab. | `visualizer/traces/live.json` and `visualizer/traces/<lesson-id>.json` |

Studio runs and lesson runs stay separate:

- Studio entry: `visualizer/studio/run.py`. Engines: `training/engines/`. Helpers: `training/common/`.
- Studio does not call `lessons/` scripts and does not overwrite lesson reports or archived traces.
- Each Studio run is a self-contained directory: data copy, trace, report, metrics, generations, adapter, or QLoRA plan.
- Chat compare on the Train page always shows base, adapter, and reloaded adapter, so you can check that the saved adapter path loads again.

| Method | Data format | Notes |
|---|---|---|
| SFT + LoRA | `instruction/input/output` JSONL | Strict-JSON ticket routing. Saves a LoRA adapter. |
| PEFT LoRA | `instruction/input/output` JSONL | A real Hugging Face causal LM plus PEFT LoRA. |
| DPO Preference | `instruction/input/chosen/rejected` JSONL | Preference objective without TRL: policy, reference, adapter. |
| QLoRA Plan | No training JSONL | A local / CUDA / VRAM / quantization plan. It does not pretend to run bitsandbytes 4-bit on Mac/MPS. |

Usage, schema, layout, and troubleshooting: [docs/08-training-studio.md](docs/08-training-studio.md). Page and API contract: [visualizer/README.md](visualizer/README.md).

To drive a lesson trace into `visualizer/traces/live.json`:

```bash
.venv/bin/python lessons/06-peft-lora/run.py --trace-delay 0.5
.venv/bin/python lessons/07-sft-baseline/run.py --trace-delay 0.5
.venv/bin/python lessons/08-dpo-preference/run.py --trace-delay 0.5
.venv/bin/python lessons/09-rlhf-reward/run.py --trace-delay 0.5
.venv/bin/python lessons/10-qlora-engineering/run.py --trace-delay 0.5
```

`--trace-delay` only slows the demo. Omit it for a normal run.

Chat Lab on the lesson page defaults to the Lesson 07 SFT baseline. Chat compare on the Train page can change model, adapter dir, and instruction, or use the latest Studio artifacts.

## The 10 lessons

| # | Topic | Report |
|---|---|---|
| 01 | Datasets pipeline | [report](lessons/01-datasets/report.md) |
| 02 | AutoTokenizer and SFT labels | [report](lessons/02-tokenizer/report.md) |
| 03 | Batch, collator, padding | [report](lessons/03-batching/report.md) |
| 04 | Minimal `transformers.Trainer` loop | [report](lessons/04-trainer/report.md) |
| 05 | Freeze the base, train, and reload a hand-written LoRA | [report](lessons/05-lora/report.md) |
| 06 | Real PEFT LoRA on `Qwen/Qwen2.5-0.5B-Instruct` | [report](lessons/06-peft-lora/report.md) |
| 07 | Strict-JSON support-ticket SFT baseline | [report](lessons/07-sft-baseline/report.md) |
| 08 | DPO on `prompt/chosen/rejected` | [report](lessons/08-dpo-preference/report.md) |
| 09 | Reward model, reference, KL, and PPO/RLHF signals | [report](lessons/09-rlhf-reward/report.md) |
| 10 | QLoRA, quantization, memory, Mac/CUDA boundary | [report](lessons/10-qlora-engineering/report.md) |

Order and checks: [lessons/README.md](lessons/README.md).

```bash
make test
.venv/bin/python lessons/run_all.py
.venv/bin/python lessons/run_all.py --quick
```

Each lesson directory contains `README.md`, `run.py`, `report.md`, `index.html`, and a local `outputs/` directory when you run it. `outputs/` is gitignored.

## Model defaults

Lessons that need a model prefer a real Hugging Face checkpoint:

- Lesson 04: `sshleifer/tiny-gpt2`, for a fast Trainer loop.
- Lesson 06: `--model-name auto` selects `Qwen/Qwen2.5-0.5B-Instruct` on the 32GB MPS machine used to write the lesson. Real PEFT LoRA.
- Lesson 07: the same `auto` rule, for the Chinese strict-JSON SFT baseline.
- Lesson 08: `auto`, a real HF causal LM for DPO.
- Lesson 09: a small local reward / KL / PPO signal. Full RLHF is not forced onto a Mac.
- Lesson 10: a hardware and training-engineering plan. It states the CUDA + bitsandbytes path and the Mac / MPS boundary.

Lesson 05 keeps a hand-written LoRA so the low-rank `A`/`B` matrices are visible. It is not a real-model training lesson.

## Reference repos

Picked from personal stars, for study:

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

On a laptop, the goal is a complete loop:

- You can explain why each field reaches the model
- You can explain which tokens contribute to loss
- You can read a training curve and an eval result
- You can save, load, and resume a checkpoint
- You can reproduce a minimal SFT + LoRA flow

Understanding a small model is worth more than launching a large model that will not run.

## Community

How to contribute: [CONTRIBUTING.md](CONTRIBUTING.md). Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

| You want to | Go here |
|---|---|
| Report a failed command with a traceback | [Issue · environment](https://github.com/harrisliangsu/llm-finetune-study/issues/new/choose) |
| Correct a doc, lesson note, or broken link | Issue · docs |
| Propose a lesson or a Studio capability | Issue · lesson / idea, or Discussions **Ideas** |
| Ask how to study, or get stuck on a concept | Discussions **Q&A** |
| Introduce yourself or post progress | Discussions **General** |
| Show a small experiment | Discussions **Show and tell** |

Welcome thread: [欢迎：自我介绍 / 学习进度 / 纠错建议](https://github.com/harrisliangsu/llm-finetune-study/discussions/1). File-level corrections belong in an issue or pull request. Open-ended talk stays in Discussions.

Repository topics already set on GitHub: `apple-silicon`, `chinese`, `chinese-llm`, `dpo`, `fine-tuning`, `finetuning`, `huggingface`, `llm`, `lora`, `machine-learning`, `peft`, `qlora`, `sft`, `study-notes`, `transformers`, `tutorial`.

## License

[MIT](LICENSE).
