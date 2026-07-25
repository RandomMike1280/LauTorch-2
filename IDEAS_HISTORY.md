# Ideas History

## Schema

| # | Idea | Form (sketch) | Score / metric | vs best | Length / cost | Parent | Branch | Status | Notes |

## Iteration 0: Baseline

| # | Idea | Form | Score | vs best | Length | Parent | Branch | Status | Notes |
|---|------|------|-------|---------|--------|--------|--------|--------|-------|
| 0 | Baseline (no model) | n/a | n/a | n/a | n/a | root | n/a | baseline | empty workspace, only docs |

## Iteration 1: Architecture decisions

| # | Idea | Form | Score | vs best | Length | Parent | Branch | Status | Notes |
|---|------|------|-------|---------|--------|--------|--------|--------|-------|
| 1 | tiny_gpt2_d16_L2 | d_model=16, n_layers=2, n_heads=4, ctx=16, vocab=96, d_ff=32, params=5888 | loss=0.72, acc=80% on 2/3 test prompts | baseline | 5888 params | 0 | arch | best-so-far | Char-level, tied embedding, pre-LN. Trained 8000 steps, batch=32, lr=5e-3 cosine. Greedy gives: "1+1 equals 2." ✓, "The capital of France is Paris." ✓, "Hey there!" ✓ |

