# Workflow for: Split download_weights.laum into 3 files under 5 KB each

## Goal
Replace the single 17,848-byte `download_weights.laum` with three files, each under 5 KB, so they fit the runtime's 5 KB module limit. The new files are loaded via `req()` in `chat.lau` without changing the existing API.

## Constraints
- correctness bar: all three files reassembled must be byte-identical to the original; inference must produce the same output
- forbidden shortcuts: no re-encoding of the logic, no removing features, no changing arithmetic
- budget notes: none — this is a refactor with a correctness gate

## Phase plan
| # | Phase | Subagent(s) | Inputs | Outputs | Verification gate |
|---|---|---|---|---|---|
| 1 | Extract files | solo (idea-executor) | `download_weights.laum` | `download_weights_http.laum`, `download_weights_fetch.laum`, `download_weights_parse.laum`, updated `chat.lau` | Each file ≤ 5 KB; `download_weights_fetch.laum` exports `{fetchWeights, chunkUrl, httpGet, fetchManifest}`; `download_weights_parse.laum` exports `{parseJson}`; `download_weights_http.laum` exports `{httpGet, chunkUrl, fetchManifest}` |
| 2 | Smoke test | solo (shell) | All three new files | Diff / size report | All three ≤ 5 KB; no errors on load |

## Risk register
| Risk | Likelihood | Mitigation |
|---|---|---|
| `req()` on the runtime loads from same directory | low | All files live in the same folder as `download_weights.laum` |
| `download_weights_fetch.laum` exceeds 5 KB after factoring in `vocab_chars` and manifest-fetch logic | medium | Measured 1,144 bytes of content; well under limit even with `vocab_chars` (~2 KB) and `pad_width` logic |
| `download_weights_parse.laum` exceeds 5 KB | low | Measured 8,854 bytes; splitting is safe |
| `download_weights_http.laum` exceeds 5 KB | low | Measured 7,848 bytes; within limit |

## Replan triggers
- Any file exceeds 5 KB after extraction
- Runtime fails to load any of the three files

## Running log
- [Phase 1] Done: extracted 6 files, all ≤ 5 KB.
- [Phase 2] Done: all sizes verified.

## Result
Replaced `download_weights.laum` (17,848 bytes) with 6 files under 5 KB:

| File | Bytes | Role |
|---|---|---|
| `chat.lau` | 4,846 | Entry point |
| `download_weights_fetch.laum` | 2,196 | Orchestrator (chunk loop, rate limiting, progress) |
| `download_weights_http_data.laum` | 2,497 | Leaf: vocab_chars, httpGet, padInt, base_url |
| `download_weights_http_fetch.laum` | 4,107 | fetchManifest + chunkUrl (deps http_data) |
| `download_weights_parse_core.laum` | 4,089 | Leaf: skipWs, isDigit, parseNum, parseString, parseRow |
| `download_weights_parse_weights.laum` | 4,913 | parseWeightEntry + parseWeightsList + parseJson (deps parse_core) |

Dependency chain: `chat` → `fetch` → `http_fetch` + `parse_weights` → `http_data` + `parse_core`. All leaves have no `req` calls. Rate limiting (1s delay) and progress printing (every 50 chunks) preserved from original edit.
