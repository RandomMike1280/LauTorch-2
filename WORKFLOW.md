# Workflow for: Lau Transformer — Correct Q&A + Fast Inference

## Goal
Train a Python Lau-Transformer language model that correctly answers all 3 test prompts, with inference running as fast as possible in Lau. Weights served via HTTP from .json files (no 5000-byte limit on weights). Inference engine in Lau with minimal per-token statements.

## Test Prompts (correctness = all must pass)
1. "Hello" → greeting response
2. "What's 1+1?" → must contain "2"
3. "What's the capital of France?" → must contain "Paris"

## Constraints
- Correctness bar: 3/3 prompts answered correctly
- Speed bar: minimize per-token statements (statement-cost 0.01s)
- No more than 5 .lau and .laum files total
- HTTP module for fetching weights from .json
- All parameters default, statement-cost can be modified during dev

## Key Leverage Points
1. **Model size**: Can scale up significantly (no .laum size limit on weights)
2. **Weight delivery**: HTTP fetch from .json for large weight files
3. **Quantization**: Keep int8 for storage, but model can be larger
4. **Architecture**: 2-layer GPT-2 style, char-level tokenizer
5. **Training data**: Synthetic Q&A pairs, heavy oversampling of test prompts
6. **Inference speed**: Optimize Lau code for minimal statements

## Phase Plan
| # | Phase | Subagent(s) | Inputs | Outputs | Verification gate |
|---|-------|-------------|--------|---------|-------------------|
| 1 | Anchor & Baseline | generalPurpose | user goal | WORKFLOW.md + IDEAS_HISTORY.md | files exist |
| 2 | Build eval harness | idea-executor + shell | chat.lau, train/ | eval script + baseline | harness runs, scores 0/3 |
| 3 | Train large model | idea-executor | train/model.py, train/train.py | weights.json via HTTP | model passes Q&A |
| 4 | Optimize Lau inference | idea-executor + brainstormer | chat.lau, weights | optimized chat.lau | same correctness, fewer statements |
| 5 | Final delivery | generalPurpose | all files | final chat.lau + weights.json | 3/3 correct, fast |

## Stopping Signals
- 3/3 prompts correct AND subsequent round shows no meaningful speed gain
- Ceiling: if model quality can't reach 3/3 even with huge model (evidence-based)
- Manual halt by user

## Risk Register
| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Model too slow to be useful | medium | Aggressive Lau optimization, statement counting |
| Model can't learn answers | low | Oversample test prompts, increase model size |
| HTTP fetch adds latency | low | Keep weights JSON reasonably sized, fetch once |
