# PLAN — Performance: tuning, batched inference, CPU budget, observability

> **Goal:** cut per-request latency on the CPU-only deployment *without*
> degrading transcription accuracy and without changing the API contract.
> The floor today is CTranslate2 inference on whisper's fixed ~30 s windows;
> the plan ranks the levers we measured, exposes them as env knobs (never
> hard-coded regressions), gates any default change behind an accuracy check,
> and adds the ops guidance + observability needed to keep the earlier
> "constant ~3.3–4.3 s" complaint from coming back (its root cause was CPU
> contention between resident models, not the server code).

## Non-goals

- No GPU / no model-size change (`base` + `int8` is the committed default).
- No `small`/`medium` benchmarking (project constraint).
- No new API fields beyond what faster-whisper already accepts.
- No decode-path micro-opts: `transcribe(path)` ≈ `transcribe(np_array)`
  (measured earlier, ~31–37 ms for PyAV decode + mel <1 ms — not the floor).

## Measured baseline (this machine, `base`/int8, CPU-only)

All runs: `jfk` ≈ 11 s (1 window), `long` = jfk×3 ≈ 33 s (2 windows), server
defaults unless noted (`condition_on_previous_text=False`):

| Config | jfk (11 s) | long (33 s) | Notes |
| --- | --- | --- | --- |
| default `beam=5/best=5`, default temp schedule | 1.37 s | 8.35 s | baseline |
| `beam=2/best=2` | 1.20 s | 4.23 s | **2.0×** on long; text ~equal |
| `beam=2/best=2` + `temperature=[0,0.2,0.4]` | 1.20 s | 3.02 s | **2.8×**; capitalization preserved |
| `BatchedInferencePipeline(batch_size=4, beam=2)` | — | **1.95 s** | **4.3×** vs baseline |
| `beam=2` + `vad_filter=True` | — | 3.36 s | VAD *adds* ~0.3 s on all-speech clips |
| `cpu_threads=1 / 2 / 4` | — | 9.31 / 5.86 / 5.51 s | saturates ≈ 4 (core count) |

Reading of the numbers:

- **Beam search is the single biggest lever** on multi-window clips (2×).
  On a single window (jfk) the win is small (~8%) because one pass dominates.
- **The default temperature schedule (`[0 … 1.0]`) is a hidden multiplier**:
  every fallback step re-runs the whole pass; trimming to 3 steps saved ~1.2 s
  on a 2-window clip *and* preserved capitalization (the plain `beam=2` run
  lost a leading "And"/"So" once).
- **Batched inference compounds**: same-quality output class as beam2, 4.3×
  vs the default on 33 s. Gains grow with clip length (better encoder
  utilization per batch); expect it to matter most for ≥1 min uploads.
- **VAD only pays when there is silence to cut** (padded recordings, meetings);
  it is a pure cost on TTS-length speech clips.
- **`cpu_threads` saturates near the physical core count** — oversubscribing
  (e.g. 8 threads × 2 concurrent requests on 4 cores) is exactly the
  contention regime that produced the reported constant 3–4 s latencies.
- Machine variance is real (a 33 s clip measured 2.14 s in an earlier probe
  vs 8.35 s today under different background load) — **benchmark relative
  ratios, not absolute seconds**, and re-measure on target hardware.

---

## Phase 0 (P0) — correctness: thread-safe lazy init of the inference pool

> Review finding, added to this plan before any tuning work: an unsynchronized
> lazy singleton is a latent thread-safety defect that would silently break the
> `WHISPER_MAX_CONCURRENT` bound this whole plan builds on — fix it first.

### Finding

`app.py` lines 526–539 — `_inference_pool()` lazily creates the module-global
`_inference_executor` with a check-then-set and **no lock**:

```python
_inference_executor: Optional[futures.ThreadPoolExecutor] = None

def _inference_pool() -> Optional[futures.ThreadPoolExecutor]:
    global _inference_executor
    if settings.max_concurrent > 0:
        if _inference_executor is None:            # check-then-set, unsynchronized
            _inference_executor = futures.ThreadPoolExecutor(
                max_workers=settings.max_concurrent, ...)
        return _inference_executor
    return None
```

**Why it does not bite today:** every current caller (`_offload`,
`_sse_stream`, `_guard_duration_ok` → `_offload`) runs on the asyncio event
loop, which is single-threaded — the init is effectively serialized.

**Why it is still a P0:** the function is a documented general-purpose
accessor, and the first future call from a *worker* thread (a very plausible
refactor — e.g. invoking `_inference_pool()` inside `_transcribe_collect` or a
streaming producer) would race:

- two threads both see `None` → **two executors created**, one leaked. The
  orphan's threads are non-daemon, so they also delay interpreter shutdown;
- the effective concurrency bound silently becomes `2 × max_concurrent` —
  exactly the oversubscription regime that caused the reported 3–4 s
  latencies, and invisible to every guard built on top of the pool;
- secondary wart: the pool snapshots `settings.max_concurrent` at first call,
  so swapping `Settings` later (tests, hot-reconfig) keeps the stale size.

### Fix (both, cheap)

1. **Eager init in `lifespan` startup** — the event loop runs init
   single-threaded at boot, deterministic and before any request can call the
   getter. Remove the lazy path entirely (the getter becomes a pure lookup;
   `None` only when `max_concurrent == 0`, i.e. unbounded-by-design).
2. **Belt-and-braces: double-checked locking** around the global
   (`threading.Lock` + re-check inside), so even a future off-loop caller
   cannot resurrect the race. A `futures.ThreadPoolExecutor` is cheap to
   create but never to leak — lock cost is nanoseconds against a ~1 s
   inference.

Also: expose `_reset_inference_pool()` (or close-and-null in lifespan
shutdown) so tests and a future config-reload path can rebuild the pool at the
new `max_concurrent` instead of inheriting a stale snapshot.

### Phase-0 tests

- unit: hammer `_inference_pool()` from 16 threads concurrently → all callers
  get the **same** executor object, and `max_workers == settings.max_concurrent`
  (would fail today under a thread-call refactor).
- unit: swap `Settings(max_concurrent=4)` + reset → pool rebuilt at 4.
- unit: `max_concurrent=0` → getter returns `None` and `_offload` uses the
  anyio pool (already covered; keep as regression).

---

## Phase 1 (P0) — inference knobs, env-exposed, defaults gated on accuracy

### A. `WHISPER_BEAM_SIZE` / `WHISPER_BEST_OF`

- `Settings` gains `beam_size: int = 5`, `best_of: int = 5` (env
  `WHISPER_BEAM_SIZE` / `WHISPER_BEST_OF`), plumbed through `_build_kwargs`.
- Default stays 5 (accuracy-first). The README gets a "latency tuning" table
  recommending `2/2` for TTS-pipeline-style short clips.
- Accuracy gate before touching the default: spot-check jfk + the existing
  Italian TTS clips at `beam=2` vs `beam=5` — exact-match/diff, plus no
  repetition loops (the known short-clip failure mode). Flip the default only
  if parity holds; otherwise ship the knob + docs.

### B. `WHISPER_TEMPERATURES` (trimmed fallback schedule)

- `Settings.temperature_schedule: str = ""` — empty = faster-whisper default;
  parse comma-separated floats (e.g. `0,0.2,0.4`) into the list faster-whisper
  accepts. Plumb via `_build_kwargs`.
- Rationale (measured): fewer fallback passes = less hidden latency, and the
  trimmed schedule *fixed* a capitalization regression seen at plain `beam=2`.
- Risk: fewer retries on genuinely garbled audio → slightly worse worst-case.
  Documented as a latency/robustness tradeoff knob, default unchanged.

### C. `WHISPER_CPU_THREADS` (right-size the pool)

- `WhisperModel(..., cpu_threads=N)`; knob default `0` = faster-whisper's own
  default (≈4 here). Guidance in README: `cpu_threads ≈ cores /
  WHISPER_MAX_CONCURRENT` keeps N concurrent transcriptions out of the
  oversubscription regime.
- Note the interplay with `WHISPER_MAX_CONCURRENT=2` (shipped): 2 × 4 threads
  on a 4-core box already oversubscribes; set one or the other, not both high.

### Phase-1 tests

- unit: `_build_kwargs` forwards the three new knobs; bad `WHISPER_TEMPERATURES`
  value falls back to default (no 500).
- e2e (real server): boot with `WHISPER_BEAM_SIZE=2 WHISPER_BEST_OF=2` — jfk
  still returns the phrase (no repetition loops) and `/health` reports the
  effective knobs.

---

## Phase 2 (P1) — batched inference for long clips (opt-in)

- `Settings.batch_size: int = 0` (`WHISPER_BATCH_SIZE`; 0 = off).
- When `> 0`, build the inference unit on
  `BatchedInferencePipeline(model=wm)` and call
  `bp.transcribe(path, batch_size=..., beam_size=..., ...)`; the SSE producer
  and `_transcribe_collect` iterate its (lazy) segment generator unchanged.
- **Semantics to document**: batched mode chunks with its own windowing and
  does not use `condition_on_previous_text`; expect minor casing/segmentation
  differences vs sequential mode (seen in the probe). Keep it opt-in for
  batch/long-audio workloads where 4× wall-time matters more than case.
- Reset-per-request: constructing `BatchedInferencePipeline` per request is
  cheap (wraps the model); cache one per store alongside `wm` if profiling
  shows otherwise.
- Interplay with `WHISPER_MAX_CONCURRENT`: batch size N × threads T is the
  real concurrency footprint — document a worked example
  (`cpu_threads=2, batch=4, max_concurrent=2` on 4 cores).

### Phase-2 tests

- unit: fake-model path still receives the right kwargs when `batch_size>0`.
- e2e (real server): `WHISPER_BATCH_SIZE=4` on the 88 s concatenated clip —
  200 with non-empty text, wall-time materially lower than the sequential
  e2e baseline (assert `< 0.75 ×` the measured sequential time to stay
  variance-safe), SSE events still stream progressively.

---

## Phase 3 (P2) — VAD, offline reload, observability, ops runbook

### D. `WHISPER_VAD` (opt-in Silero VAD pre-pass)

- `Settings.vad: bool = False` → `vad_filter=True` in `_build_kwargs`.
- Only worth it for inputs with leading/trailing silence or long pauses;
  measured a ~10% *penalty* on all-speech TTS clips, so default stays off and
  the README says when to flip it.

### E. No network on evicted reload

- `WhisperModel` construction can hit the HF hub for a metadata check even on
  a warm cache; after an idle eviction this turns a ~0.5 s warm reload into a
  multi-second stall when the network is flaky. Set
  `local_files_only=True` (respect an opt-out env if someone really wants
  hub checks), and document `HF_HUB_OFFLINE=1` for air-gapped deploys.
- Test: with the knob set, evict + request → reload succeeds with no outbound
  call (assert via env `HF_HUB_OFFLINE=1` in a real-server e2e).

### F. Per-stage timings (make future tuning measurable)

- Log (debug) or expose at `/health` rolling p50/p95 for: upload write,
  duration check, language resolution, inference, total. Cheap
  `time.perf_counter()` brackets around the existing offloaded chunks.
- Purpose: the earlier "constant 3–4 s" complaint took an investigation to
  root-cause; with stage timings the next one is a glance.

### G. Ops runbook (README section)

- **One resident model per host** (or CPU-quota the containers): the measured
  3–4 s floor under 4-model contention is an ops problem, not a code bug —
  say so explicitly with the numbers.
- Worked sizing example: 4 cores → `WHISPER_MAX_CONCURRENT=2`,
  `WHISPER_CPU_THREADS=2`, `WHISPER_BATCH_SIZE=0` (or `=4` with
  `MAX_CONCURRENT=1` for long-file workloads).
- Reminder that `WHISPER_IDLE_UNLOAD_S` trades RAM (~450 MB) for a ~0.5 s
  warm reload — keep hot for latency-critical deployments.

### H. Reproduce-the-baseline env presets (README + docker-compose)

> Every row of the "Measured baseline" table must map 1:1 to a copy-paste env
> block, so an operator can reproduce any measured case without reverse-
> engineering knobs from prose.

- README gets a **"Reproducing the measured baseline"** subsection under the
  latency-tuning table: one commented env block per row of the table
  (baseline row = "set nothing", stated explicitly). Presets (knobs land in
  Phases 1–2; docs ship with each knob, table completed in Phase 3H):

| Baseline row | Env preset |
| --- | --- |
| default `beam=5/best=5` | *(no env — server defaults)* |
| `beam=2/best=2` | `WHISPER_BEAM_SIZE=2` `WHISPER_BEST_OF=2` |
| beam2 + trimmed schedule | `WHISPER_BEAM_SIZE=2` `WHISPER_BEST_OF=2` `WHISPER_TEMPERATURES=0,0.2,0.4` |
| Batched bs4 beam2 | `WHISPER_BEAM_SIZE=2` `WHISPER_BEST_OF=2` `WHISPER_BATCH_SIZE=4` |
| beam2 + VAD | `WHISPER_BEAM_SIZE=2` `WHISPER_BEST_OF=2` `WHISPER_VAD=true` |
| `cpu_threads=1 / 2 / 4` | `WHISPER_CPU_THREADS=1` (or `2`, `4`) |

- Each preset must also appear as a commented block in `docker-compose.yml`
  (same comment style as the existing knob blocks), tagged with the expected
  relative speedup from the table (e.g. "≈2.8× on 33 s clips vs defaults").
- Footnote the caveats next to the presets: beam2 gated on the Phase-1
  accuracy check; batched mode's casing/segmentation drift; VAD is a
  regression on all-speech clips; machine-variance note (ratios, not
  absolute seconds).
- **Drift guard test** (same philosophy as the `/health` field assertions): a
  unit test parses `README.md` and `docker-compose.yml` and asserts every
  `WHISPER_*` knob referenced in the preset table appears in both files —
  so knobs and docs cannot silently diverge.

---

## Config surface (all new, all optional)

| Env | Default | Meaning | Phase |
| --- | --- | --- | --- |
| `WHISPER_BEAM_SIZE` | `5` | Beam width (2 ≈ 2× on multi-window clips; gate on accuracy) | 1 |
| `WHISPER_BEST_OF` | `5` | Sampling candidates when `temperature > 0` | 1 |
| `WHISPER_TEMPERATURES` | *(fw default)* | Comma-separated fallback schedule, e.g. `0,0.2,0.4` | 1 |
| `WHISPER_CPU_THREADS` | *(fw default)* | CT2 intra-op threads (`cores / max_concurrent` guidance) | 1 |
| `WHISPER_BATCH_SIZE` | `0` | `BatchedInferencePipeline` batch (0 = sequential) | 2 |
| `WHISPER_VAD` | `false` | Silero VAD pre-pass (only for silence-heavy inputs) | 3 |
| `WHISPER_HF_OFFLINE` | `false` | `local_files_only` on (re)load; no hub round-trip | 3 |

Each phase updates: `app.py` (+ `/health` reporting of effective knobs),
README env table + "latency tuning" section, `docker-compose.yml` commented
knobs, and both suites. Phases stay individually green and ship separately.

## Risks / open questions

- **Phase 0 first:** every concurrency guarantee in Phases 1–2 leans on the
  bounded pool; shipping tuning on top of an unsynchronized pool init would
  compound the latent race instead of removing it.
- **Accuracy tradeoffs are real but small at `beam=2`** (one capitalization /
  article drop observed on jfk×3 at plain `beam=2`); Phase 1's accuracy gate
  decides whether the default moves or only the knob ships.
- **Batched mode changes segmentation/casing** (no `condition_on_previous_text`)
  — opt-in only, documented, never a silent default flip.
- **Benchmark variance across machines** is large (2.1 s vs 8.4 s for the same
  33 s clip under different load); all shipped numbers must be ratios from a
  single run-set, re-measured on target hardware before publishing SLAs.
- **Batched × threads × concurrent** is a 3-dimensional CPU budget; the README
  example in Phase 3G is the guardrail, and `/health` stage timings (F) are
  the feedback loop.
