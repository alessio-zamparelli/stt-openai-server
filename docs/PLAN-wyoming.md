# PLAN — Wyoming ASR protocol

## Goal

Expose the server as a Wyoming speech-to-text provider so Home Assistant can use
it alongside (or instead of) the official Whisper app. The listener is opt-in and
shares the HTTP API's model, idle eviction, language rules, resource limits, and
inference serialization.

## Design decisions

1. **In-process asyncio listener.** `stt_api/services/wyoming.py` uses
   `wyoming.server.AsyncTcpServer` on uvicorn's event loop. No second process and
   no second copy of the model. `AsyncServer.start()` binds without blocking and
   is stopped from the FastAPI lifespan.
2. **Package-based module boundary.** The listener lives in the `stt_api`
   package and imports `stt_api.runtime` normally, so launching via
   `python app.py` (the compatibility launcher) cannot create a duplicate copy
   of `settings`/`store`/locks.
3. **Opt-in.** `STT_WYOMING=false` by default. `docker-compose.yml` enables it
   explicitly on `:10300`.
4. **Final transcript only.** `supports_transcript_streaming=false`. This matches
   the Home Assistant Whisper app and the reference `wyoming-faster-whisper`
   batch backend.
5. **External VAD.** `requires_external_vad=true`; Home Assistant sends
   `audio-stop` when the command ends. `transcribe.vad_sensitivity` is ignored.
6. **One utterance per connection.** After `transcript`, the server returns
   `False` from the handler and closes the connection, matching the reference
   server.
7. **Reuse the HTTP pipeline.** The handler normalizes audio to 16 kHz mono s16,
   writes a WAV into a `SpooledTemporaryFile`, then calls the same
   `_resolve_language`, `_build_runner`, `_build_kwargs`, and
   `_transcribe_collect` helpers through `runtime._offload`.
8. **Optional zeroconf.** `STT_WYOMING_ZEROCONF=true` registers
   `_wyoming._tcp.local.` with a MAC-derived or configured name. Registration
   failures are logged and ignored; manual host/port still works.

## Protocol mapping

| Wyoming event | Server behavior |
| --- | --- |
| `describe` | Reply `info` with one `asr` program named `stt-api` and model `STT_MODEL_NAME`. Does not load the model or touch the idle timer. |
| `select-program` | Ignored. |
| `transcribe` | Store language; ignore `name`; touch the model store. |
| `audio-start` | Reset the buffer/converter. |
| `audio-chunk` | Convert to 16 kHz mono s16, enforce limits, append to WAV. |
| `audio-stop` | Transcribe, send `transcript`, close the connection. |
| unknown | Ignored; connection stays open. |

## Error behavior

Wyoming ASR has no error event. Empty audio, overflow, decode failure, and
timeout all produce `Transcript(text="")`; the failure is logged and counted in
`/health.wyoming.errors`.

## Limits

- `STT_MAX_AUDIO_SECONDS` and `STT_MAX_UPLOAD_MB` are checked incrementally while
  chunks arrive. Once exceeded, the buffer is freed and the rest of the stream is
  drained until `audio-stop`.
- `STT_REQUEST_TIMEOUT_S` wraps the inference wait. CTranslate2 cannot be
  preempted, so a timed-out decode continues in the worker until it finishes.
- `STT_MAX_CONCURRENT` bounds queued Wyoming and HTTP work; `_inference_lock`
  still serializes decodes on the one shared model.

## Observability

`/health.wyoming` reports `enabled`, bind host/port, zeroconf state, the mDNS
name, and counters (`requests`, `errors`, `active`). `/health.latency_ms`
records `language`, `inference`, and `wyoming` (audio-stop → transcript) stages.

## Tests

- `tests/test_wyoming.py`: real TCP listener + official Wyoming client, fake
  model. Covers `describe`/`info`, round-trip transcription, language
  normalization/fallback, non-16 kHz conversion, empty/oversized audio, ignored
  events, connection close, store activity, and zeroconf registration.
- `tests/e2e_wyoming_test.py`: real server/model, streams `jfk.flac` as PCM and
  asserts the transcript.
- Documentation drift guard covers the five `STT_WYOMING*` knobs.

## Non-goals

- Interim `transcript-start` / `transcript-chunk` / `transcript-stop` streaming.
- Wyoming translation events (the HTTP translations endpoint remains).
- Authentication/encryption: Wyoming is designed for a trusted network.
- Name/term prompt biasing (`transcript_names` / `transcript_terms`); the server
  prompt remains `STT_INITIAL_PROMPT`.
