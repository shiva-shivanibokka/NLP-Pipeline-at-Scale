# Continuous Deployment — Cloud Run (one-time setup)

`.github/workflows/deploy.yml` auto-deploys the API to Cloud Run whenever backend files
(`api/`, `src/`, `configs/`, `Dockerfile`, `requirements-api.txt`) change on `main`.
It needs two GitHub secrets. Set them up once:

## 1. Create a deploy service account + key (run locally, gcloud is authenticated)

```bash
PROJECT=search-ranking-system            # your GCP project id
SA=cloud-run-deployer

gcloud iam service-accounts create $SA --project $PROJECT \
  --display-name "GitHub Actions Cloud Run deployer"

EMAIL="$SA@$PROJECT.iam.gserviceaccount.com"

# Roles needed for `gcloud run deploy --source` (build + deploy)
for ROLE in roles/run.admin roles/cloudbuild.builds.editor \
            roles/artifactregistry.admin roles/storage.admin \
            roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding $PROJECT \
    --member "serviceAccount:$EMAIL" --role "$ROLE" --quiet
done

# Download a JSON key (this is a secret — don't commit it)
gcloud iam service-accounts keys create key.json --iam-account $EMAIL
```

## 2. Add the secrets to GitHub

Repo → **Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Value |
|---|---|
| `GCP_SA_KEY` | the entire contents of `key.json` |
| `GCP_PROJECT` | `search-ranking-system` |

Then delete the local key: `rm key.json`.

## 3. Done

Push any change under `api/`, `src/`, `configs/`, `Dockerfile`, or `requirements-api.txt`
to `main` and the API redeploys automatically. You can also trigger it manually from the
**Actions** tab (`workflow_dispatch`).

> **Security note:** a long-lived JSON key is the simplest option. For a hardened setup,
> switch to Workload Identity Federation (keyless) — same workflow, replace the
> `credentials_json` auth step with `workload_identity_provider` + `service_account`.
