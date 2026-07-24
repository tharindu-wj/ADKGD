#!/usr/bin/env bash
# Push training artifacts to S3 for durability. A g4dn.xlarge's local NVMe and a
# TERMINATED spot instance both lose their disk; an EBS root survives a STOP but
# not a terminate. Run this after each job (or on a cron/timer) so a checkpoint
# is never one interruption away from gone.
#
#   ./sync_out.sh s3://my-bucket/adkgd        # or set ADKGD_S3=s3://my-bucket/adkgd
#
# Needs AWS creds: attach an instance IAM role with s3:PutObject on the bucket
# (best), or run `aws configure` once.
set -euo pipefail

BUCKET="${1:-${ADKGD_S3:-}}"
[[ -n "$BUCKET" ]] || { echo "usage: sync_out.sh s3://your-bucket/prefix  (or set ADKGD_S3)"; exit 2; }
cd "${PROJECT_DIR:-$HOME/ADKGD}"

# Checkpoints (all *.pt, including per-epoch snapshots) and every job log.
aws s3 sync experiments/kgsage/outputs/checkpoints "$BUCKET/checkpoints" \
    --exclude '*' --include '*.pt'
aws s3 sync . "$BUCKET/logs" --exclude '*' --include '*.out.txt'
# ADKGD matrix artifacts (per-run JSON + logs) live under checkpoints/<dataset>/.
[[ -d checkpoints ]] && aws s3 sync checkpoints "$BUCKET/adkgd_checkpoints"

echo "Synced *.pt + *.out.txt (+ ADKGD checkpoints/) to $BUCKET"
