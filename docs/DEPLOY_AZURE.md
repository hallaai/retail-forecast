# Deploying the forecast demo to Azure, cheaply

Status: **ready container image definition + GitHub Actions build. Not yet deployed or
tested on a live Azure account.** Prices are approximate (late 2026); check the Azure
pricing calculator before relying on them.

## What gets deployed

One container runs a single `uvicorn` process that serves:
- the static dashboard (`web/` + the committed `web/data/*.json`), which works on its own, and
- the per-series API (`/api/series/{id}`), which reads three precomputed parquet files baked
  into the image (`test_forecasts.parquet`, `panel.parquet`, `calendar.parquet`, ~92 MB).

There is **no database and no LLM** — the heavy modelling (`scripts/run_pipeline.py`) runs
offline; the container only reads its outputs. So the runtime image excludes torch,
lightgbm, chronos, etc. (see `server/requirements.txt`).

A browser sign-in box (one shared password) sits in front of everything except `/healthz`.

## Recommendation

Run the container on **Azure Container Apps, Consumption plan, scaled to zero**, with
`APP_PASSWORD` in front of it. The image is **pulled from GitHub Container Registry (GHCR)**;
nothing is stored in an Azure registry.

| Piece | Choice | Cost when idle | Cost per use |
|---|---|---|---|
| App | Container Apps, Consumption, 0.5 vCPU / 1 GiB, min 0 / max 1 replica | $0 (scaled to zero) | per vCPU-second / GiB-second while running; the monthly free grant (~180,000 vCPU-s, 360,000 GiB-s, 2 M requests per subscription) covers light use |
| Image registry | **GHCR (free for a public repo)** — image built by GitHub Actions and pulled by Azure | $0 | $0 |
| Build | In the cloud — GitHub Actions. No local Docker. | $0 | negligible |
| Logs | Container Apps env with logs destination `none` (or Log Analytics, first GBs free) | $0 | $0 |
| HTTPS | Azure-managed certificate on `*.azurecontainerapps.io` | $0 | $0 |

With scale-to-zero, the bill is essentially a few seconds of container time per visit, most
of it inside the free grant.

## Architecture

```
browser ──HTTPS──> Container Apps ingress (TLS ends here, HTTP→HTTPS redirect)
                       │ internal network, plain HTTP
                       ▼
                   retail-forecast container (uvicorn, port 8000)
                       │  password gate → FastAPI
                       ├── static dashboard (web/ + web/data/*.json)
                       └── /api/series/{id}  ← 3 parquet files baked into the image
```

The image comes from GHCR; GitHub only builds and stores it. The running app and its URL
come from Azure: `https://<APP>.<random>.<region>.azurecontainerapps.io`.

## Authentication

**Built in: one shared password (HTTP Basic).**
- `APP_PASSWORD` (12+ chars) and optional `APP_USER` (default `analyst`). The browser shows
  its own sign-in box once per session. The value lives in `web/.env` locally and is passed
  to Azure as a secret — it is **never** committed or baked into the image.
- The image sets `AUTH_REQUIRED=1`, so a container without `APP_PASSWORD` refuses to start
  rather than running open.
- After `AUTH_MAX_FAILURES` (10) wrong attempts within `AUTH_LOCKOUT_SECONDS` (300) from one
  address, that address gets 429 until the window passes.
- `/healthz` is open for probes and returns only `{"ok": true}`. The API docs pages
  (`/docs`, `/openapi.json`) are off.
- Limits: one shared password, no per-user audit trail, no logout (close the browser).
  Rotate by updating the secret.

The password rides in the `Authorization` header of every request. That is safe **only**
over HTTPS, so never expose the app on plain HTTP (keep `--allow-insecure false`).

**Upgrade (optional, no code): Container Apps built-in auth ("Easy Auth").** Microsoft Entra
ID / GitHub / Google sign-in in front of the app, per-user identity, MFA and logout, at no
extra cost. Enable with `az containerapp auth …` and either keep the password gate as a
second layer or turn it off (`AUTH_REQUIRED=0`, no `APP_PASSWORD`).

## No Docker on your machine

You do **not** need Docker Desktop. The image is built in the cloud by GitHub Actions and
pulled by Azure from GHCR.

Prerequisite: the Azure CLI (`az`) with the `containerapp` extension.
`az login`, then `az extension add --name containerapp --upgrade`.

### Step 1 — build the image (GitHub Actions → GHCR)

1. Push to GitHub. The workflow `.github/workflows/image.yml` runs (on changes to
   `server/`, `src/m5demo/`, `web/`, the 3 data files, or the Dockerfile) or can be run by
   hand from the **Actions** tab. It builds from the repo root and pushes
   `ghcr.io/hallaai/retail-forecast:<sha>` and `:latest`. No local Docker.
2. Make the package pullable by Azure: either make the GHCR package **public**
   (Packages → retail-forecast → Package settings → Change visibility → Public), or create a
   token with `read:packages` and pass it as the registry password in Step 2.

### Step 2 — create the Azure app (pulls from GHCR)

```bash
RG=retail-forecast-rg; LOC=swedencentral; ENV=retail-forecast-env; APP=retail-forecast
az group create -n $RG -l $LOC
az containerapp env create -n $ENV -g $RG -l $LOC --logs-destination none

# The password — type it so it stays out of shell history (it is in web/.env locally)
read -rs APP_PW

az containerapp create -n $APP -g $RG --environment $ENV \
  --image ghcr.io/hallaai/retail-forecast:latest \
  --target-port 8000 --ingress external \
  --cpu 0.5 --memory 1.0Gi --min-replicas 0 --max-replicas 1 \
  --secrets app-password="$APP_PW" \
  --env-vars APP_PASSWORD=secretref:app-password APP_USER=analyst AUTH_REQUIRED=1
# Private GHCR package only: add pull credentials
#   --registry-server ghcr.io --registry-username hallaai --registry-password "$GHCR_READ_TOKEN"

az containerapp ingress update -n $APP -g $RG --allow-insecure false
az containerapp show -n $APP -g $RG --query properties.configuration.ingress.fqdn -o tsv
```

Open the printed URL, sign in as `analyst` with the password, and the dashboard loads.

### Redeploy after a code change

The workflow only **builds and pushes** the image; it does not touch the running app. After
a push to `main` has built a new `:<sha>`, point Container Apps at it. Pin the **full commit
SHA**, not `:latest` — a moving tag does not force a pull, so the app would keep serving the
old revision.

```bash
# the SHA the Actions run built (git rev-parse HEAD, or the Actions/Packages page)
az containerapp update -n retail-forecast -g retail-forecast-rg \
  --image "ghcr.io/hallaai/retail-forecast:<full-sha>" --only-show-errors
```

Check which image is actually running:

```bash
az containerapp show -n retail-forecast -g retail-forecast-rg \
  --query "properties.template.containers[0].image" -o tsv
```

Rotate the password:

```bash
az containerapp secret set -n retail-forecast -g retail-forecast-rg --secrets app-password=<new>
az containerapp revision restart -n retail-forecast -g retail-forecast-rg \
  --revision "$(az containerapp revision list -n retail-forecast -g retail-forecast-rg --query '[0].name' -o tsv)"
```

## What to expect

- **Cold start.** The first request after scaling to zero waits for the container to start
  and the parquet store to load into memory (the `@lru_cache` in `server/app.py`). Estimated
  at 10–30 s, not yet measured. Set `--min-replicas 1` to avoid it, at the price of an
  always-on replica.
- **Scale-in.** After the last request the replica stays up for the cooldown (~5 min default)
  before scaling to zero.
- **One replica** keeps the sign-in lockout counters (process-local memory) exact.

## Not done yet

- Not deployed; no Azure account was available while this was written. The image definition
  was validated by running the server from the same file layout locally (fail-closed start,
  `/healthz`, sign-in challenge, UI + API served) and by the test suite (`tests/test_auth.py`,
  `tests/test_api.py`). The image itself is first built by the Actions workflow.
- No infrastructure-as-code (Bicep/Terraform) yet; the steps above are CLI commands.
- Cold-start time and real monthly cost are unmeasured.
