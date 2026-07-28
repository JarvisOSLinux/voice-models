# Contributing to voice-models

This guide explains how to add a speech model to the registry.

It works like the AUR or Homebrew: you fork, add one directory, and open a pull
request. CI validates the format automatically; a maintainer reviews the facts.

The one rule that matters more than any formatting convention:

> **Every fact in an entry must be something you verified, not something you
> read.** URLs, hashes, sizes, licenses, sample rates, and above all the
> `platforms` list. If you could not verify it, do not ship it — an unverified
> entry is worse than a missing one, because a missing entry is obviously
> missing.

---

## Table of contents

1. [Before you start](#before-you-start)
2. [Step 1 — Verify the model](#step-1--verify-the-model)
3. [Step 2 — Fork and branch](#step-2--fork-and-branch)
4. [Step 3 — Create the model directory](#step-3--create-the-model-directory)
5. [Step 4 — Write model.json](#step-4--write-modeljson)
6. [Step 5 — Add the index entry](#step-5--add-the-index-entry)
7. [Step 6 — Sync and validate](#step-6--sync-and-validate)
8. [Step 7 — Open the PR](#step-7--open-the-pr)
9. [Updating an existing entry](#updating-an-existing-entry)
10. [Removing an entry](#removing-an-entry)
11. [FAQ](#faq)

---

## Before you start

- The model must be **publicly downloadable from a stable URL** — a release
  asset or a pinned Hugging Face revision, not a "latest" redirect and not a
  link that expires.
- You must be able to **actually run it** on at least one of `linux`, `darwin`,
  `windows`. That run is what earns the entry its `platforms` list.
- You need Python 3 (stdlib only) to run the scripts. No other dependencies.

---

## Step 1 — Verify the model

Do this first. Everything else is transcription.

**1. Download every artifact and hash it.**

```bash
curl -fL -O https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-lessac-medium.tar.gz
sha256sum voice-en-us-lessac-medium.tar.gz
stat -c '%s' voice-en-us-lessac-medium.tar.gz     # sizeBytes (macOS: stat -f %z)
```

**2. Run it.** Load the model in its engine and produce real output — synthesize
a sentence, transcribe a clip, score a frame. Note the **engine version** you
used; it goes in `notes`.

```bash
mkdir unpacked && tar -xzf voice-en-us-lessac-medium.tar.gz -C unpacked
echo 'Registry verification.' | piper -m unpacked/en-us-lessac-medium.onnx -f out.wav
soxi out.wav        # confirm non-trivial, non-silent audio — and read the real sample rate
```

**3. Measure the cost.**

```bash
du -sm unpacked   # diskMb — unpacked model files only, excluding the archive
/usr/bin/time -v <your run command> 2>&1 | grep 'Maximum resident'   # ramMb
```

Extract into a directory of its own and measure *that*. `diskMb` is the unpacked
size (`docs/model-reference.md` § requirements); measuring the working directory
counts the downloaded archive and the output WAV too, which roughly doubles it.

Approximate is fine and expected — say how you measured it in `notes`.

**4. Establish the license.** This is where entries most often go wrong.

- Do **not** infer the model's license from the repository's `LICENSE` file. A
  repo is very often Apache-2.0 or MIT for the *code* while the shipped weights
  are under something else. `openwakeword.hey-jarvis` is exactly that: the repo
  is Apache-2.0, the pre-trained models are CC-BY-NC-SA-4.0, and only the README
  says so.
- Check, in this order: the model card shipped **inside** the download, the
  upstream README's section on the models, then the dataset the model was trained
  from.
- When upstream declares no SPDX identifier, use a `LicenseRef-` id (e.g.
  `LicenseRef-Blizzard2013-Lessac`) and put what upstream *does* say in `notes`.
  Do not round a vague license up to a permissive one.

**5. Read the real audio parameters** off the produced or expected audio, not off
the documentation. The Piper seed entry records 16000 Hz because that is what the
shipped config and every produced WAV actually are, while the upstream model card
says 22,050 Hz.

---

## Step 2 — Fork and branch

```bash
git clone https://github.com/<your-username>/voice-models.git
cd voice-models
git checkout -b add/<model-id>
```

---

## Step 3 — Create the model directory

The directory name **is** the model id:

```
models/
  piper.en-us-lessac-medium/
    model.json          ← the only file
```

Ids are lowercase and dotted, `<engine>.<model-name>`, matching
`^[a-z0-9][a-z0-9-]*(\.[a-z0-9][a-z0-9-]*)+$` — both halves take a hyphen, since
two of the engines do (`faster-whisper`, `whisper-cpp`):

- `piper.en-us-lessac-medium`
- `openwakeword.hey-jarvis`
- `vosk.en-us-small-0.15`
- `faster-whisper.small`

Keep the upstream model's own name in the second half. If upstream calls it
`small`, call it `small`; do not editorialize it into `whisper-small-int8`.

**One capability, one entry.** A model that both transcribes and diarizes is one
entry with two `capabilities`. A bundle that is really two models — a recognizer
and a separate speaker model — is two entries.

---

## Step 4 — Write model.json

Full field reference: [`docs/model-reference.md`](docs/model-reference.md).

```json
{
  "id": "piper.en-us-lessac-medium",
  "name": "Piper Lessac (medium, en-US)",
  "summary": "Single-speaker U.S. English VITS neural text-to-speech voice for the Piper engine, trained from scratch on the Blizzard 2013 Lessac corpus.",
  "version": "0.0.2",
  "kind": "tts",
  "engine": "piper",
  "languages": ["en-us"],
  "capabilities": ["streaming"],
  "platforms": ["linux"],
  "requirements": { "diskMb": 61, "ramMb": 191, "accelerator": "cpu" },
  "license": {
    "spdx": "LicenseRef-Blizzard2013-Lessac",
    "url": "https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html",
    "notes": "Upstream declares no SPDX identifier: …"
  },
  "trustStatus": "community",
  "manifest": "https://raw.githubusercontent.com/JarvisOSLinux/voice-models/main/models/piper.en-us-lessac-medium/model.json",
  "upstream": {
    "project": "https://github.com/OHF-Voice/piper1-gpl",
    "homepage": "https://github.com/rhasspy/piper"
  },
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
  },
  "audio": { "sampleRateHz": 16000, "channels": 1, "sampleWidthBits": 16 },
  "notes": "Exercised on Linux x86_64 with piper-tts 1.6.0 + onnxruntime 1.28.0 on CPU: …"
}
```

Four things to get right:

- **`manifest`** is the manifest's own raw URL on `main`. It must point at
  `models/<your-id>/model.json` — the validator cross-checks it against where the
  index fetches the file from.
- **`trustStatus` is `community`.** Always, for a new entry. Promotion to
  `official` is a maintainer action and CI will reject a self-promotion.
- **`platforms` is what you ran it on.** Not what upstream claims. One OS is a
  perfectly good entry; someone else can widen it later in a PR of their own.
- **List every artifact the model needs to run.** The openWakeWord entry carries
  three files because the engine otherwise expects its preprocessor models in
  its own package directory, put there by a separate unverified
  `openwakeword.utils.download_models()` call. "Fetch this entry" should be a
  complete, offline-able operation.

---

## Step 5 — Add the index entry

Add the identity fields to the `models` map in `registry.json`. Everything else
is filled in for you:

```json
"piper.en-us-lessac-medium": {
  "id": "piper.en-us-lessac-medium",
  "name": "Piper Lessac (medium, en-US)",
  "summary": "Single-speaker U.S. English VITS neural text-to-speech voice …",
  "version": "0.0.2",
  "kind": "tts",
  "engine": "piper",
  "trustStatus": "community",
  "manifest": "https://raw.githubusercontent.com/JarvisOSLinux/voice-models/main/models/piper.en-us-lessac-medium/model.json"
}
```

Do **not** hand-write `languages`, `capabilities`, `platforms`, `requirements`,
`license`, or `integrity`. Those are mirrored from your manifest by the sync
script, and a hand-edited mirror is exactly the failure the validator exists to
catch.

Nor any other field. An entry carries its identity, its `manifest` URL, the
mirrored fields and `integrity` — nothing else. Anything extra fails validation
and `sync_registry.py` deletes it, because a field only the index has is one no
reviewer ever read.

---

## Step 6 — Sync and validate

```bash
python3 scripts/sync_registry.py     # mirrors your manifest into the index, computes the hash
python3 scripts/validate_registry.py # the same gate CI runs
python3 scripts/selftest_registry.py # only if you touched scripts/
```

`sync_registry.py` prints what it changed. Expect to see your manifest hash and
one line per mirrored field. Commit the resulting `registry.json` diff along with
your manifest — CI runs `sync_registry.py --check` and fails if you forgot.

If the validator complains about a mirrored field, fix the **manifest** and
re-run sync. Never fix it by editing `registry.json`.

---

## Step 7 — Open the PR

Title: `Add <model-id>`.

The PR body is where your verification evidence goes — commands and their real
output. That is what a maintainer reviews; the diff itself is mostly transcription.

```markdown
## piper.en-us-lessac-medium

**Downloaded and hashed**
$ sha256sum voice-en-us-lessac-medium.tar.gz
d45ca50084c0558eb9581cd7d26938043bc8853513da47c63b94d95a2367a5c9  voice-en-us-lessac-medium.tar.gz
$ stat -c '%s' voice-en-us-lessac-medium.tar.gz
58197843

**Ran it** — Linux x86_64, piper-tts 1.6.0, onnxruntime 1.28.0, CPU
$ echo 'Registry verification.' | piper -m en-us-lessac-medium.onnx -f out.wav
$ soxi out.wav
Channels: 1  /  Sample Rate: 16000  /  Duration: 00:00:05.89

**License** — MODEL_CARD names the Blizzard 2013 Lessac dataset license; upstream
declares no SPDX id, so the entry uses LicenseRef- and records that in notes.

**Platforms** — ["linux"], earned by the run above. Not tested on macOS/Windows.
```

Keep the evidence in the PR, not in the repo. `notes` carries the durable summary
("exercised with piper-tts 1.6.0 on CPU"); the transcript belongs in the review.

CI then runs the self-test, the validator, and the sync check. All three must pass.

---

## Updating an existing entry

Edit the manifest, re-run `sync_registry.py`, and say in the PR body what you
re-verified.

Three changes get an automatic review annotation and deserve explicit evidence:

- **A widened `platforms` list.** This field records what the registry actually
  exercised, so adding `darwin` means *you ran it on macOS* — show that.
- **A changed `license.spdx`.** A relicense reaches everyone who already deployed
  the model. Link to where upstream states the new terms.
- **A changed artifact `url` or `sha256`** (or `huggingface` `repo`/`revision`).
  That pair decides which bytes everyone downloads next; paste the `sha256sum`
  of the file you actually fetched.

If the artifacts themselves changed — a new upstream release — bump `version` and
replace the hashes, having downloaded and run the new files. Do not edit a hash
without re-downloading. A digest that moves while `version` stays put gets its
own annotation asking why, so say so in the PR body if there is a good reason.

---

## Removing an entry

Two stages, safest first:

1. **Soft revoke.** Set `trustStatus` to `"removed"` in the manifest and re-run
   sync. Consumers refuse new installs, existing ones keep working, and the
   entry's provenance stays intact. Use `"deprecated"` instead when the model is
   simply superseded rather than unsafe.
2. **Hard removal**, after a grace period: delete the `models/<id>/` directory
   **and** the `registry.json` entry in the same PR. The validator fails a
   half-done removal in either direction — a dangling entry whose manifest is
   gone, or an orphan directory nothing references.

---

## FAQ

**Can I add a model I did not train?**
Yes — that is the normal case. The registry catalogs models, it does not host or
claim them. What you are asserting is that you fetched and ran *these bytes*, and
that the license is what you say it is.

**The engine downloads its own weights. What do I put in `source`?**
The `huggingface` shape, with a pinned `revision`. See
[`docs/model-reference.md`](docs/model-reference.md#source). The revision is
mandatory: it is the only thing recording which weights were vetted.

**My model needs an engine that is not in the enum.**
Add it to `ALLOWED_ENGINES` in `scripts/validate_registry.py` and document its
packaging in the engine table in `docs/model-reference.md`, in the same PR as the
first model that uses it. Unknown engines are a hard error on purpose: `engine`
is what tells a consumer how to load the files.

**My model has a capability that is not in the list.**
Use it — an unknown capability is a warning, not an error — and add it to the
vocabulary table in `docs/model-reference.md` in the same PR.

**Do I need to compute hashes for a `huggingface` source?**
No. There are no per-file hashes there; the engine handles the download. Pin the
revision instead.

**Why doesn't CI verify my hashes by downloading the files?**
Because a hash CI computed would be whatever the server served that day, which
verifies nothing. The hash is meaningful precisely because a human asserted "this
is the file I ran." See
[`docs/REGISTRY-GUIDE.md`](docs/REGISTRY-GUIDE.md#integrity-philosophy).

**Can I mark my own entry `official`?**
No. CI compares `trustStatus` against the base branch and fails any promotion
without a maintainer's `trust-approved` label.

**Does the registry's AGPL-3.0 license apply to my model?**
No. AGPL-3.0 covers this repository — the scripts, the docs, the metadata. Each
model carries its own license in its `license` field, and the registry does not
relicense anything it catalogs.
