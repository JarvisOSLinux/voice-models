# Model Reference

`model.json` is the per-model manifest. It lives at `models/<id>/model.json` and
is the **source of truth** for everything the registry knows about a model.

`registry.json` is the index. It repeats — mirrors — the fields a consumer needs
in order to *choose* a model, so browsing the catalog costs one HTTP request
instead of one per model. The mirror is written only by
`scripts/sync_registry.py`; `scripts/validate_registry.py` fails any PR where the
two disagree. Never hand-edit a mirrored field in `registry.json`.

```
models/piper.en-us-lessac-medium/model.json   ← you write this
registry.json                                 ← sync_registry.py writes this
```

---

## Table of contents

1. [Identity](#identity)
2. [Top-level fields](#top-level-fields)
3. [kind](#kind)
4. [engine](#engine)
5. [languages](#languages)
6. [capabilities](#capabilities)
7. [platforms](#platforms)
8. [requirements](#requirements)
9. [license](#license)
10. [trustStatus](#truststatus)
11. [upstream](#upstream)
12. [source](#source)
13. [audio](#audio)
14. [notes](#notes)
15. [integrity](#integrity-index-only)
16. [Worked example](#worked-example)

---

## Identity

A model id is lowercase and dotted: `<engine>.<model-name>`.

```
^[a-z0-9][a-z0-9-]*(\.[a-z0-9][a-z0-9-]*)+$
```

Examples: `piper.en-us-lessac-medium`, `openwakeword.hey-jarvis`,
`vosk.en-us-small-0.15`, `faster-whisper.small`.

Both halves take a hyphen. The engine half has to, because `faster-whisper` and
`whisper-cpp` carry one in their names — an id pattern that rejected them would
block two of the five engines this registry already vouches for.

Four things must agree, and the validator enforces all four:

| | |
|---|---|
| the `registry.json` map key | `"piper.en-us-lessac-medium": { … }` |
| the entry's `id` | `"id": "piper.en-us-lessac-medium"` |
| the manifest's `id` | `"id": "piper.en-us-lessac-medium"` |
| the directory name | `models/piper.en-us-lessac-medium/` |

The manifest also states its own canonical URL in `manifest`, which must match
the URL the index fetches it from. That one catches a manifest copy-pasted out of
another model's directory, where every other field looks plausible.

---

## Top-level fields

| Field | Type | Required | Mirrored into `registry.json` | Description |
|-------|------|----------|-------------------------------|-------------|
| `id` | string | Yes | — (it is the map key) | Model id, see [Identity](#identity). |
| `name` | string | Yes | Yes | Human-readable display name. |
| `summary` | string | Yes | Yes | One line: what this model is and what it does. |
| `version` | string | Yes | Yes | Upstream's version for these weights. Not semver-enforced — upstream numbering wins (`"0.0.2"`, `"0.1"`, `"0.15"`). |
| `kind` | string | Yes | Yes | `stt` / `tts` / `wake`. See [kind](#kind). |
| `engine` | string | Yes | Yes | The runtime that loads these files. See [engine](#engine). |
| `languages` | array | Yes | Yes | BCP-47-ish codes, or `["multilingual"]`. |
| `capabilities` | array | Yes | Yes | Open vocabulary. See [capabilities](#capabilities). |
| `platforms` | array | Yes | Yes | What this registry **vetted**, not what upstream claims. |
| `requirements` | object | Yes | Yes | Approximate disk/RAM/accelerator cost. |
| `license` | object | Yes | Yes | `spdx` + `url`, optional `notes`. |
| `trustStatus` | string | Yes | Yes | `community` / `official` / `deprecated` / `removed`. |
| `manifest` | string | Yes | Yes (as the fetch URL) | The manifest's own canonical raw URL. |
| `upstream` | object | Yes | No | Where the model comes from. |
| `source` | object | Yes | No | How to fetch it. See [source](#source). |
| `audio` | object | No | No | Input/output audio contract. Expected for `stt` and `wake`. |
| `notes` | string | No | No | Caveats, and what verification was actually performed. |

`integrity` appears **only** in the index entry — a manifest cannot contain its
own hash.

The index entry carries exactly the fields marked *Mirrored: Yes* above, plus
`id`, `manifest` and `integrity`. That is a closed set the validator enforces:
any other key in an entry is an error, and `sync_registry.py` removes it. A
field that exists only in the index was reviewed by nobody, since the reviewer
reads the manifest and the consumer reads the index. If a consumer needs
something new, add it to the manifest **and** to `SYNCED_FIELDS`, in the same
PR.

---

## kind

| Value | Meaning |
|-------|---------|
| `stt` | Speech to text (transcription). |
| `tts` | Text to speech (synthesis). |
| `wake` | Wake-word / keyword spotting: scores short audio frames for one phrase. |

`kind` is the coarse filter a consumer applies first ("give me the TTS voices").
A model that does two of these is two entries, one per capability, because a
consumer loads them into different parts of its pipeline.

---

## engine

The runtime that can load these files. An unknown engine is a **validation
error**, not a warning: `engine` is what tells a consumer how to unpack and load
the artifacts, so an entry naming an engine no consumer implements is a catalog
row nobody can act on.

| Value | Packaging it expects |
|-------|----------------------|
| `vosk` | A zip that unpacks to a model directory. |
| `faster-whisper` | CTranslate2 weights, fetched by the engine from Hugging Face. |
| `whisper-cpp` | A single GGML/GGUF file. |
| `piper` | An ONNX voice plus its `.onnx.json` config. |
| `openwakeword` | ONNX (or TFLite) classifier plus the shared preprocessor models. |

**The list grows by PR.** Adding an engine means adding it to `ALLOWED_ENGINES`
in `scripts/validate_registry.py` and documenting its packaging in this table, in
the same PR as the first model that uses it.

---

## languages

```json
"languages": ["en-us"]
```

Lowercase BCP-47-ish tags: `en`, `en-us`, `tr`, `zh-cn`. Use the literal
`["multilingual"]` for a model that is not usefully described by a list (Whisper's
multilingual checkpoints, for instance).

Prefer the most specific tag the model actually justifies: `en-us` for a voice
trained on U.S. English, plain `en` for a wake-word model that does not care
about the accent.

---

## capabilities

```json
"capabilities": ["streaming"]
```

v1 vocabulary:

| Value | Meaning |
|-------|---------|
| `streaming` | Can consume/produce audio incrementally, without buffering the whole utterance. |
| `word-timestamps` | Emits per-word timing. |
| `punctuation` | Restores punctuation and casing. |
| `speaker-diarization` | Labels who spoke when. |
| `voice-cloning` | Can imitate a target speaker from a reference sample. |
| `multi-speaker` | Ships more than one speaker in one model file. |
| `gpu-acceleration` | Has a supported GPU execution path. |

An unrecognized capability is a **warning**, not an error — the vocabulary is
meant to grow, and an unknown value degrades to "not advertised" rather than
breaking a download. If you introduce one, add it to this table in the same PR.

---

## platforms

```json
"platforms": ["linux"]
```

Allowed values: `linux`, `darwin` (macOS), `windows`. Required, non-empty.

**`platforms` records what this registry actually exercised.** Not what upstream
claims, not what "should work". If nobody has run the model on macOS, `darwin`
does not appear in the list — even if upstream's README says it is supported.
This is the same semantics `mcp-registry` uses for its `platforms` field, and it
exists so that "the registry vouches for this" means something specific.

Widening the list is therefore a new claim about work someone did. A PR that adds
a platform to an existing entry gets an automatic review annotation saying so.

---

## requirements

```json
"requirements": { "diskMb": 61, "ramMb": 191, "accelerator": "cpu" }
```

| Field | Type | Meaning |
|-------|------|---------|
| `diskMb` | positive int | Unpacked size on disk, in MiB. |
| `ramMb` | positive int | Rough peak resident memory while running. |
| `accelerator` | enum | `cpu` (runs fine on CPU), `cuda` (needs NVIDIA GPU), `any` (either). |

**These numbers are approximate and documented as such.** They exist so a
consumer can rule models in or out for a constrained device — a Raspberry Pi
budget, a container memory limit — not so it can compute an exact allocation.
Record how you measured them in `notes`; the seed entries do.

They still have to be *numbers*: a stringified `"61"` fails validation, because a
consumer that cannot compare `diskMb` across entries cannot do the one thing this
field is for.

---

## license

```json
"license": {
  "spdx": "CC-BY-NC-SA-4.0",
  "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
  "notes": "Non-commercial only: …"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `spdx` | Yes | SPDX identifier. When upstream declares none, use a `LicenseRef-` id (e.g. `LicenseRef-Blizzard2013-Lessac`) rather than guessing at an equivalent. |
| `url` | Yes | Where the actual license text lives. |
| `notes` | No | One line naming the restrictions a consumer will care about. |

**License is required, and this is not bureaucracy.** Speech models — TTS voices
above all — routinely ship under dataset licenses with real restrictions:
non-commercial clauses, research-only clauses, share-alike obligations. Both seed
entries carry one. A catalog that omits the license is handing its consumers a
legal problem disguised as a convenience.

Two rules for filling it in:

- **Do not infer the model license from the repository's `LICENSE` file.** A repo
  is very often Apache-2.0 for the *code* while the shipped weights are under
  something else entirely. `openwakeword.hey-jarvis` is exactly that case: the
  repo is Apache-2.0, the pre-trained models are CC-BY-NC-SA-4.0, and the second
  fact is only stated in the README.
- **When upstream is vague, say so in `notes`** instead of rounding to a
  permissive identifier. `piper.en-us-lessac-medium` records that the dataset
  license text was not retrievable during verification — that is a more useful
  fact for a consumer than a confident guess.

Changing the `spdx` of an existing entry raises a review annotation: a relicense
reaches everyone who already deployed the model.

---

## trustStatus

| Value | Meaning |
|-------|---------|
| `community` | Submitted and validated, not maintainer-vetted. The default for everything new. |
| `official` | A maintainer verified the entry end to end. **Requires the `trust-approved` PR label** — the validator fails a self-promotion. |
| `deprecated` | Superseded; still fetchable, should not be chosen for new work. |
| `removed` | Revoked. Consumers should refuse to install it; the entry stays for provenance. |

Revoking is two stages, safest first: set `trustStatus` to `removed` (the entry
and its provenance stay intact), and only later delete the entry *and* its
`models/<id>/` directory. The validator's orphan-directory check blocks a
half-done removal in either direction.

---

## upstream

```json
"upstream": {
  "project": "https://github.com/OHF-Voice/piper1-gpl",
  "homepage": "https://github.com/rhasspy/piper"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `project` | Yes | The maintained project this model belongs to. |
| `homepage` | No | Where a human should read about it — a model card, docs page, or the historical repo when the project has moved. |

`source` says where the bytes are; `upstream` says who is behind them. They are
frequently not the same URL: Piper's voices are still published as release assets
on the archived `rhasspy/piper` repo while development continues in
`OHF-Voice/piper1-gpl`.

---

## source

Exactly **one** of two shapes. The validator rejects a source carrying keys from
both, because a consumer would have to guess where the bytes come from.

### `artifacts` — files fetched and verified by the consumer

```json
"source": {
  "type": "artifacts",
  "items": [
    {
      "url": "https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-lessac-medium.tar.gz",
      "sha256": "d45ca50084c0558eb9581cd7d26938043bc8853513da47c63b94d95a2367a5c9",
      "sizeBytes": 58197843,
      "format": "tar.gz",
      "unpack": true
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `url` | https URL | Direct download. Must be stable — a release asset, not a "latest" redirect. |
| `sha256` | 64 lowercase hex | **Required.** The consumer verifies this after download. |
| `sizeBytes` | positive int | Exact size. Lets a consumer show progress and refuse an obviously wrong response. |
| `format` | enum | `zip`, `tar.gz`, `onnx`, `tflite`, `json`, `bin`. |
| `unpack` | bool | Whether the consumer should extract it after verifying. |

**Multiple items are normal.** `openwakeword.hey-jarvis` has three: the wake-word
classifier plus openWakeWord's two shared preprocessor models, which the engine
otherwise expects to find in its own package directory, put there by a separate
unverified `openwakeword.utils.download_models()` call. List every file the
model needs to run, so that "fetch this entry" is a complete, offline-able
operation.

### `huggingface` — weights the engine fetches itself

```json
"source": {
  "type": "huggingface",
  "repo": "Systran/faster-whisper-small",
  "revision": "<commit sha>",
  "engineManaged": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `repo` | `org/name` | The Hugging Face repo. |
| `revision` | string | **Required.** A commit sha (or tag) pinning what was vetted. |
| `engineManaged` | bool | `true` — the engine, not the consumer, performs the download. |

There is no per-file hash here, because the consumer never sees the individual
files: `faster-whisper` resolves and caches them internally. `revision` is
therefore doing all of the provenance work, which is why it is mandatory even
though the engine will happily default to `main`.

---

## audio

```json
"audio": {
  "sampleRateHz": 16000,
  "channels": 1,
  "sampleFormat": "int16",
  "frameSamples": 1280,
  "frameMs": 80
}
```

The audio contract: what the model consumes (`stt`, `wake`) or produces (`tts`).
`sampleRateHz` is required whenever the block is present; the remaining keys are
an open set, because engines describe frames differently. Known numeric keys —
`channels`, `sampleWidthBits`, `frameSamples`, `frameMs` — are type-checked when
present.

An `stt` or `wake` entry with no `audio` block gets a warning. It is recoverable,
but it is exactly the field whose absence makes a consumer guess a sample rate,
and a wrong guess produces silent garbage rather than an error.

**Record what you measured, not what the docs say.** The Piper seed entry
documents 16000 Hz because that is what the shipped config and every produced WAV
actually are, while upstream's model card says 22,050 Hz. The measured value is
the one a consumer's resampler needs.

---

## notes

One free-text field, and the most useful one in the manifest. Use it for:

- **what was actually verified, with versions** — "Exercised in-session on Linux
  x86_64 with piper-tts 1.6.0 + onnxruntime 1.28.0 on CPU";
- **where upstream's documentation is wrong**, and what the measurement was;
- **how the artifacts fit together** — which of them are required together, what
  the tarball unpacks to;
- **what was deliberately left out** — when a variant is not cataloged, say
  whether it was skipped for lack of exercise or tried and rejected.

The last one matters more than it looks: a reader who finds a gap should be able
to tell "nobody got to this yet" from "this was tried and rejected".

---

## integrity (index only)

```json
"integrity": { "manifestSha256": "…" }
```

SHA-256 of the manifest file's bytes, recomputed by `scripts/sync_registry.py`
and never edited by hand. It binds the index entry to the exact manifest a
reviewer approved — see [`REGISTRY-GUIDE.md`](REGISTRY-GUIDE.md) §
"Integrity philosophy".

---

## Worked example

`models/openwakeword.hey-jarvis/model.json`, complete:

```json
{
  "id": "openwakeword.hey-jarvis",
  "name": "openWakeWord hey jarvis",
  "summary": "Wake-word model that scores 80 ms frames of 16 kHz mono audio for the spoken phrase \"hey jarvis\", run through openWakeWord's ONNX pipeline.",
  "version": "0.1",
  "kind": "wake",
  "engine": "openwakeword",
  "languages": ["en"],
  "capabilities": ["streaming"],
  "platforms": ["linux"],
  "requirements": { "diskMb": 4, "ramMb": 256, "accelerator": "cpu" },
  "license": {
    "spdx": "CC-BY-NC-SA-4.0",
    "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "notes": "Non-commercial only: …"
  },
  "trustStatus": "community",
  "manifest": "https://raw.githubusercontent.com/JarvisOSLinux/voice-models/main/models/openwakeword.hey-jarvis/model.json",
  "upstream": {
    "project": "https://github.com/dscripka/openWakeWord",
    "homepage": "https://github.com/dscripka/openWakeWord/blob/v0.5.1/docs/models/hey_jarvis.md"
  },
  "source": {
    "type": "artifacts",
    "items": [
      { "url": "…/hey_jarvis_v0.1.onnx", "sha256": "94a13cfe…", "sizeBytes": 1271370, "format": "onnx", "unpack": false },
      { "url": "…/melspectrogram.onnx", "sha256": "ba2b0e0f…", "sizeBytes": 1087958, "format": "onnx", "unpack": false },
      { "url": "…/embedding_model.onnx", "sha256": "70d16429…", "sizeBytes": 1326578, "format": "onnx", "unpack": false }
    ]
  },
  "audio": {
    "sampleRateHz": 16000,
    "channels": 1,
    "sampleFormat": "int16",
    "frameSamples": 1280,
    "frameMs": 80
  },
  "notes": "All three ONNX artifacts are required together: …"
}
```

Then run `python3 scripts/sync_registry.py`, which writes the index entry and its
`integrity` block for you.
