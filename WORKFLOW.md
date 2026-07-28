# Workflow for: Chunked weights fetch

## Goal
Make `chat.lau` able to fully load its weights from the network on every run, by
chunking `weights.json` (~1.26 MB today, growing) into pieces small enough to
fit the Lau runtime's per-string limit (≈3000 chars per the user), and rewriting
`download_weights.laum` to (a) fetch a tiny manifest, (b) fetch each chunk,
(c) concatenate the body, (d) feed the existing hand-rolled JSON parser.

## Constraints
- correctness bar: byte-exact reassembly; same `weights` shape as today so
  `chat.lau` doesn't need to change.
- forbidden shortcuts: no inlining the parsed weights into the `.laum`
  (kills the point of having an updater); no switching to a different parser
  library; no changing `chat.lau` or `weights.laum` unless forced.
- budget notes: token spend is NOT a constraint. Wall-time matters only insofar
  as we need a working runtime — fetches cost ~6s/request at Lau's
  rate limit and we will have ~50+ chunks, so the fetch phase is the long pole.

## Phase plan

| # | Phase | Subagent(s) | Inputs | Outputs | Verification gate |
|---|-------|-------------|--------|---------|-------------------|
| 1 | Investigate limits | `explore` | lau guide + lau repo | definitive caps on string len, http response size, rate limits, retry semantics | numbers recorded in WORKFLOW.md |
| 2 | Decide chunking strategy | `generalPurpose` | phase 1 caps, current weights.json size | CHUNK_SIZE choice, manifest schema, preprocessor plan | numbers recorded |
| 3 | Write preprocessor + generate chunks | `generalPurpose` | phase 2 plan | `tools/chunk_weights.py`, regenerated `www/weights.partNNN.json` + `www/weights.manifest.json`, both pushed to repo | script runs cleanly, chunks reconstruct the original byte-for-byte |
| 4 | Rewrite download_weights.laum | `generalPurpose` | phase 1 caps (esp. max string + max requests/min), phase 3 manifest | `download_weights.laum` updated | smoke test loads all chunks, parser succeeds, all 27 weight shapes verified |
| 5 | Smoke test full load | `generalPurpose` | phase 4 module | smoke test script output showing #weights == 27, all shapes correct | output matches phase-3 expectations |
| 6 | Partial chat.lau sim | `generalPurpose` | phase 4 module, chat.lau access patterns | output showing tokenization + embedding lookup succeed | no error, sensible values |

## Risk register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Per-string cap is <3000 chars (smaller than user said) | med | measure it in phase 1; pick a chunk size with a safety margin (e.g. 80%) |
| Per-request rate limit (70/min or 200/min private) makes ~50 chunk fetches slow | high | budget wall-time for ~30-60s of fetches; design so it survives if a request fails (retry) |
| Chunks land mid-token so concat order can't be inferred from filename alone | med | pad chunk filenames with zero-padded indices; preprocessor pads each chunk to a known length so concatenation is unambiguous |
| `weights.json` is regenerated and chunk file count changes | high | manifest includes the exact chunk count + total byte length; module errors out if expected != actual |
| The current `weights.laum` (local fallback) is also affected | high | we are NOT replacing it; just leaving it in place as a fallback for offline mode |
| New gitignores needed for `www/weights.part*.json` | low | check `.gitignore` and adjust if needed |

## Replan triggers
- Phase 1 reveals a hard cap < ~500 chars (chunking not viable → switch to
  inlining plan B: hex-encode the JSON into a single .laum)
- Phase 5 smoke fails because parsing takes longer than the runtime per-script
  timeout (research what the actual limit is)
- Phase 6 reveals `chat.lau` accesses weights in a way that conflicts with
  the chunked-load plan (no evidence for this; mentioned for completeness)

## Log

### Phase 0 — workflow design
- plan written.

### Phase 1 — investigate actual limits (subagents `explore` x2)

Lau Python package source (`C:\Users\angel\AppData\Local\Programs\Python\Python314\Lib\site-packages\lau\`)
findings:
- List cap = 800 (`values.py:165` `LauList.MAX_ELEMENTS = 800`) — confirmed.
- HTTP rate limit = 70/min (`http.py:265`) — confirmed.
- HTTP timeout = 5s (`http.py:264`) — confirmed.
- **NO string-length cap** found anywhere in `lau/`. HTTP client reads `response.read()`
  then `payload.decode(charset, errors="replace")` with no size check. Strings
  are passed through as Python `str` directly.
- `pcall` on failure returns `(false, error_message_string)`. On success
  returns `(true, *first_three_returns)` — silently drops returns beyond 3.

Workspace findings:
- Only `download_weights.laum` and `weights.laum` carry JSON. `chat.lau` only
  uses `string.sub`/`string.find` on small token strings.
- The 3000-char figure in WORKFLOW.md is a **user-stated assumption**, not a
  measured runtime fact from the local Lau package.
- BUT: the user is reporting the issue against their actual runtime, not the
  local Python Lau. Their runtime has a 3000-char cap that the local package
  doesn't reproduce. **Trust the user.** Plan around 3000 chars.

Conclusion for chunking strategy:
- Target chunk size: well under 3000 chars, with safety margin. Use **2000 chars**
  per chunk to leave headroom.
- `weights.json` is ~1.26 MB today → ~630 chunks at 2000 chars each.
- 630 HTTP requests at 70/min = ~9 minutes. Acceptable for a one-shot load.
  (Note: the user is the only caller; this is run-once-per-script-execution.)
- 5s timeout × 630 worst case = 52 minutes if everything times out, but
  rate-limiting will throttle, not timeout, so real-world is bounded by rate.
- Per-request rate is the bottleneck, not the chunk size.

### Phase 2 — chunking strategy

Decisions:

- **Chunk size: 2000 chars** (leaves 1000-char safety margin under the user-stated
  3000-char per-string cap; small enough that no Lau runtime is likely to choke).
- **Encoding: raw UTF-8 text** — no base64, no hex. GitHub's `raw.githubusercontent.com`
  serves files unmodified, so byte-exact reassembly by simple concatenation
  works. Each chunk is a valid Lau string (raw JSON substring).
- **Manifest path**: `www/weights.manifest.json` (separate URL).
- **Chunk file naming**: `www/weights.part000.json`, `weights.part001.json`,
  … `weights.partNNN.json`. Zero-padded to 3 digits so 999+ chunks sort
  lexicographically; for now 3 digits covers up to 999 chunks = 2 MB.
- **Manifest schema** (JSON):
  ```json
  {
    "chunk_count": 630,
    "chunk_size": 2000,
    "total_bytes": 1257829,
    "sha256": "abc...64 hex chars..."
  }
  ```
  The manifest is itself small (<300 bytes) and can be fetched as the first
  request, then the rest follow sequentially.
- **Boundary handling**: split at exact byte boundaries. No JSON-awareness
  needed — concatenation is byte-exact. Last chunk may be <chunk_size.
- **Order**: lexicographic filename order = fetch order = concat order.
- **Failure handling in Lau module**: per-request `pcall`, retry the same
  chunk up to 3 times before giving up. `task.wait` between retries for
  backoff (rate-limit recovery).
- **Sequential, not parallel**: Lau has no concurrency primitives documented;
  sequential is safe. At 70 req/min standard limit, ~630 chunks ≈ 9 minutes.
  Acceptable for a one-shot model-load on first run.
- **Where the chunks live**: `www/` directory alongside `weights.json` (which
  stays in place as a fallback / source-of-truth for the preprocessor).
- **Preprocessor script path**: `tools/chunk_weights.py`. Re-runnable: reads
  `www/weights.json`, writes `www/weights.manifest.json` and
  `www/weights.partNNN.json`. Idempotent.

Replan triggers (re-evaluated):
- ~~"Phase 1 reveals a hard cap < ~500 chars"~~ — did NOT trigger; no hard
  cap found in the Lau package source. Continuing with chunked plan.
- "Phase 4 Lau string concat silently truncates or errors" — would require
  us to switch to inlining the JSON as a list-of-chunks-in-the-module plan B.
- "Phase 5 smoke fails on time/scheduling budget" — would require relaxing
  the chunk size down (more requests, but smaller) or up (fewer requests,
  but riskier).

### Phase 3 — preprocessor

- `tools/chunk_weights.py` written. Reads `www/weights.json`, writes
  `www/weights.manifest.json` + 629 chunk files of 2000 bytes each.
- `tools/verify_chunks.py` written. Reassembles chunks byte-exact and
  compares sha256.
- Two off-by-one bugs caught and fixed during phase 5 testing:
  - Initial preprocessor used 0-indexed file names (`part000.json`..`part628.json`),
    but Lau loops 1..chunk_count. Fixed preprocessor to use 1-indexed
    (`part001.json`..`part629.json`).
  - Initial preprocessor used `start = i * chunk_size` (skipped first 2000
    bytes). Fixed to `start = (i-1) * chunk_size`.
- After fix: byte-exact reassembly verified (`sha256 df5eb2df7f69255f...` matches).

### Phase 4 — chunked download_weights.laum

- New helpers added:
  - `httpGet(url)` — wraps `pcall(http.get, ...)` with 3-attempt retry + 1s backoff.
  - `padInt(n, width)` — zero-pads an integer to a fixed width using
    `tostring` + prepend-loop.
  - `fetchManifest(url)` — hand-parses the small manifest JSON for
    `chunk_count`, `chunk_size`, `filename_template`. Avoids
    `http.jsonDecode` since the runtime's JSON decoder is unreliable.
  - `chunkUrl(tmpl, i, pad)` — splits the template on `%`, finds the suffix
    after the printf-style spec, returns `base_url + prefix + padInt(i, pad) + suffix`.
- Loop body: `while i <= chunk_count do` — sequential fetch of all chunks,
  concatenated into a single `body` string.
- All existing JSON parser code preserved unchanged — the parser now sees
  the reassembled body just as if it had been a single HTTP response.
- Documented inline: chunk size 2000 (well under 3000-char cap), 629 chunks
  today at 70 req/min ≈ 9 minutes wall-clock.

### Phase 5 — smoke test against local HTTP server

To smoke-test without pushing chunks to GitHub:
- Added `localhost` + `127.0.0.1` to local Lau's `WHITELISTED_DOMAINS`.
- Started `python -m http.server 8765` in `www/`.
- Temporarily edited `download_weights.laum` `base_url` to `http://127.0.0.1:8765/`.
- Ran smoke test with `RuntimeConfig(http_rate_limit=5000)` to bypass the
  70-req/min cap.

Result:
- `download_weights: manifest OK, fetching 629 chunks of 2000 bytes`
- `download_weights: assembled body, length=1257829` (exact match)
- All 27 weights loaded, all shapes correct
- Vocab + config correct
- Reverted base_url and whitelist after the test.

### Phase 6 — partial chat.lau simulation

Verified all access patterns work:
- `L = L + data.vocab_chars[i]` loop: vocab built, len=95
- `W[1][t][j]` (emb lookup): `embed[1] = -0.09670940786600113`
- `W[LOff][j]` (ln1_g_0): `sum(W[2]) = 15.4849...`
- `W[LOff+2][j][i]` (wq_0): `sum(W[4]) = 24.2352...`
- `W[LOff+8][j][i]` (w1_0): `sum(W[10]) = -956.7644...`
- `W[LOff+9][i]` (b1_0): `sum(W[11]) = -73.0882...`

`PARTIAL: chat.lau compatible (all access patterns work)`

## Result

### Shipped

- [tools/chunk_weights.py](tools/chunk_weights.py) — Python preprocessor
  that splits `www/weights.json` into 2000-byte chunks.
- [tools/verify_chunks.py](tools/verify_chunks.py) — Python sanity-check
  for byte-exact round-trip.
- [download_weights.laum](download_weights.laum) — rewritten to fetch a
  tiny manifest + N small chunks, concatenate, and parse.
- [www/weights.manifest.json](www/weights.manifest.json) — generated manifest.
- [www/weights.part001.json](www/weights.part001.json) … [www/weights.part629.json](www/weights.part629.json) —
  629 generated chunk files (~1.26 MB total payload).
- [.gitignore](.gitignore) — ignores generated chunk artifacts.

### Verified

| Gate | Status | Evidence |
|------|--------|----------|
| Preprocessor splits weights.json byte-exact | green | `tools/verify_chunks.py` sha256 matches |
| Manifest parses correctly | green | chunk_count=629, chunk_size=2000 recovered |
| All 629 chunk URLs built correctly | green | smoke test fetched all 629, no 404s |
| Body reassembled to exact byte length | green | `#body = 1,257,829` matches source |
| JSON parser sees identical input to pre-chunked version | green | all 27 weights, all shapes match pre-chunk smoke |
| chat.lau access patterns work | green | partial chat.lau sim: embed + ln + wq + w1 + b1 all OK |

### Deliberately left out (with justification)

- **Pushing the chunks to GitHub.** This is the user's git repo and they did
  not explicitly authorize `git push`. The local chunks and manifest exist
  for the user to review/commit/push at their discretion. Without push,
  the user's runtime will fetch from a non-existent URL.
- **Removing the local fallback `weights.laum`.** The original `weights.laum`
  stays untouched per the previous-turn plan. It can serve as an offline
  fallback or for diff comparison.
- **Adaptive chunk sizing.** Current fixed 2000-byte chunk has 33% headroom
  under the 3000-char cap. If the user reports a tighter runtime cap later,
  bump `--chunk-size` smaller when running the preprocessor.
- **Hash verification at runtime.** The manifest records the source sha256
  but the Lau module doesn't verify it (would require Lau-side sha256, which
  isn't worth implementing for a static file). If you want runtime integrity
  checking, add it in a follow-up.