#!/usr/bin/env bash
set -euo pipefail

echo "🧪 DRY RUN MODE - Testing deployment workflow"
echo "This script simulates the deployment process without actually deploying"

GCP_PROJECT_ID=kggen-ai
CLOUD_RUN_SERVICE=app-test
CLOUD_RUN_REGION=us-central1
ARTIFACT_REGISTRY_REPO=app
IMAGE_NAME=${IMAGE_NAME:-cloud-run-app-test}
IMAGE_TAG=${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%s)}
ARTIFACT_REGISTRY_LOCATION=${ARTIFACT_REGISTRY_LOCATION:-$CLOUD_RUN_REGION}
IMAGE_URI="${ARTIFACT_REGISTRY_LOCATION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

printf "[deploy-test] Would build and push image %s\n" "$IMAGE_URI"

# Change to project root for build context
cd "$(dirname "$0")/.."

echo "[deploy-test] Current directory: $(pwd)"
echo "[deploy-test] Checking if Dockerfile exists..."

if [[ -f "app/Dockerfile" ]]; then
    echo "app/Dockerfile found"
else
    echo "app/Dockerfile not found"
    exit 1
fi

echo "[deploy-test] Would create symlinks..."
echo "[deploy-test] Would run: gcloud builds submit --tag \"$IMAGE_URI\" --project \"$GCP_PROJECT_ID\" ."

deploy_args=(
  --image "$IMAGE_URI"
  --region "$CLOUD_RUN_REGION"
  --project "$GCP_PROJECT_ID"
  --platform managed
  --allow-unauthenticated
  --quiet
)

if [[ -n "${CLOUD_RUN_SERVICE_ACCOUNT:-}" ]]; then
  deploy_args+=(--service-account "$CLOUD_RUN_SERVICE_ACCOUNT")
  echo "[deploy-test] Would use service account: $CLOUD_RUN_SERVICE_ACCOUNT"
fi

if [[ -n "${CLOUD_RUN_MAX_INSTANCES:-}" ]]; then
  deploy_args+=(--max-instances "$CLOUD_RUN_MAX_INSTANCES")
  echo "[deploy-test] Would set max instances: $CLOUD_RUN_MAX_INSTANCES"
fi

if [[ -n "${CLOUD_RUN_MIN_INSTANCES:-}" ]]; then
  deploy_args+=(--min-instances "$CLOUD_RUN_MIN_INSTANCES")
  echo "[deploy-test] Would set min instances: $CLOUD_RUN_MIN_INSTANCES"
fi

if [[ -n "${CLOUD_RUN_CPU:-}" ]]; then
  deploy_args+=(--cpu "$CLOUD_RUN_CPU")
  echo "[deploy-test] Would set CPU: $CLOUD_RUN_CPU"
fi

if [[ -n "${CLOUD_RUN_MEMORY:-}" ]]; then
  deploy_args+=(--memory "$CLOUD_RUN_MEMORY")
  echo "[deploy-test] Would set memory: $CLOUD_RUN_MEMORY"
fi

if [[ -n "${CLOUD_RUN_CONCURRENCY:-}" ]]; then
  deploy_args+=(--concurrency "$CLOUD_RUN_CONCURRENCY")
  echo "[deploy-test] Would set concurrency: $CLOUD_RUN_CONCURRENCY"
fi

printf "[deploy-test] Would deploy service %s to Cloud Run with args:\n" "$CLOUD_RUN_SERVICE"
printf "  %s\n" "${deploy_args[@]}"

echo "[deploy-test] Dry run completed successfully!"
echo "[deploy-test] All checks passed - the real deployment should work"