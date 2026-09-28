# Security

## Reporting a problem

Please report security problems privately, not in a public issue: email
**info@mycomap.org** with "MycoMap Vision security" in the subject, or use
GitHub's private vulnerability reporting on this repository. Include what you
found, how to reproduce it, and what it affects. We aim to reply within a week.

## What this repository holds, and what it doesn't

The code, the evaluation method and the web app are here. The data is not, and
never will be:

- **Photos** belong to their photographers on iNaturalist. Many are licensed for
  non-commercial use only, and some are all rights reserved (used for training
  only while permission is sought). They are stored privately, never in this repo.
- **The manifest, embeddings, contributor lists and trained models** are derived
  from those photos and stay private until the photographers' permissions allow
  otherwise.
- **Locations.** Records can hold the true coordinates of observations that are
  obscured on iNaturalist, often for rare or sensitive species. No part of this
  project may publish them or make them recoverable.
- **Settings and credentials** (database routes, bucket names, cloud keys) come
  from the environment or a git-ignored `.env`, never from the code.

## The public API

`mv serve` limits what a caller can make it do: at most 10 photos and 80 MB per
request, 25 MB and 40 megapixels per photo (checked before decoding), a request
rate per address (`MV_RATE_LIMIT`), a cap on identifications waiting for the GPU
(`MV_MAX_QUEUE`), and only approved backbones (`MV_ALLOWED_BACKBONES`). It only
reads its data. When it runs behind a proxy, configure the proxy to pass the real
client address (and uvicorn's `--proxy-headers` with a trusted `--forwarded-allow-ips`)
so the per-address limit applies to callers, not to the proxy.

## Dependencies

Python dependencies are pinned with hashes in `requirements/*.lock`
(regenerate with `pip-compile --generate-hashes`); the web app is pinned by
`web/pnpm-lock.yaml`. Backbone weights come from timm and open_clip via the
Hugging Face hub; prefer safetensors weights, and only backbones on the allow-list
are loaded by the public API.
