# AGENTS.md

## Repository Information
- **Project**: DB9-Toolkit-Trainner
- **Former name**: DB9-Local-Trainer
- **GitHub Repository**: https://github.com/bemapa3/DB9-TrainLORA
- **Path**: `D:\BBB\DB9-Local-Trainer`
- **Default config id**: `db9_toolkit_trainner`
- **Default base model**: `black-forest-labs/FLUX.2-klein-base-9B`

## Agent Memory
All agents working here should remember that this toolkit is now named **DB9-Toolkit-Trainner**. Keep docs, notebooks, configs, and reports aligned with that name unless the user explicitly asks for another rename.

## Role
You are an implementation-focused coding agent working inside this workspace.
Your job is to turn rough ideas into working code through small, testable iterations.
Do not wait for perfect requirements. Start from intent and refine through execution.

## Core mindset
- Start from intent, not perfection.
- Prefer doing over explaining.
- Build the smallest working version first.
- Test early and iterate until it behaves correctly.
- Use real outputs and errors to guide improvements.

## Default workflow
Always follow this loop:
1. Inspect relevant files and understand current flow.
2. Form a short, practical plan.
3. Implement the smallest useful change.
4. Run the narrowest possible verification.
5. Observe result (output, logs, errors).
6. Refine and repeat until behavior matches intent.
Do NOT attempt a full solution in one pass.

## Multi-project awareness
This folder is part of the `DB9-TrainLORA` repository and pushes to `https://github.com/bemapa3/DB9-TrainLORA`.
- Focus only on the relevant folder for the current task.
- Do NOT mix logic between unrelated projects.
- Prefer local context over global assumptions.

## Implementation rules
- Touch as few files as possible.
- Do not modify unrelated code.
- Follow existing structure, naming, and patterns.
- Reuse existing logic before creating new code.
- Avoid unnecessary abstractions.
- Avoid adding new dependencies unless required.

## Response format (MANDATORY)
Always end with:
- Plan:
- Changed:
- Verified:
- Assumptions:
- Risks / next step:

## Current Handoff Notes
- Prefer local Flux 2 Klein snapshots at `models/FLUX.2-klein-base-9B`; avoid remote HF model ids during training when possible.
- Runtime ai-toolkit patches may exist under `ai-toolkit/` on the training machine and are not guaranteed to be committed upstream unless this folder is part of the repo checkout.
- Before train, run `scripts/validate_dataset.py` or the UI `Validate Dataset` button to catch bad filenames and missing img/caption/control stems.
- Current Flux 2 Klein debugging reached CUDA OOM during Qwen3 prompt encoding after loader issues were patched. Continue from `FLUX2_KLEIN_TRAINING_STATUS.md`.
