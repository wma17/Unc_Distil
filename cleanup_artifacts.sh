#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN="${DRY_RUN:-1}"

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

remove_path() {
    local path="$1"
    if [ ! -e "$path" ]; then
        return
    fi
    if [ "$DRY_RUN" = "1" ]; then
        log "DRY RUN remove: $path"
    else
        rm -rf -- "$path"
        log "Removed: $path"
    fi
}

log "Repository root: $ROOT"
log "DRY_RUN=$DRY_RUN"

# Keep:
# - TinyImageNet/checkpoints_rich_fake_ood_12m        (current retrain workspace)
# - TinyImageNet/checkpoints_fake_ood_ft_wd005_blr1e3 (best complete TinyImageNet run)
# - MNIST/checkpoints, MNIST/checkpoints_fake_ood, CIFAR-10/checkpoints, CIFAR-10/checkpoints_16members

stale_dirs=(
    "$ROOT/TinyImageNet/checkpoints_fake_ood_ft"
    "$ROOT/TinyImageNet/checkpoints_fake_ood_ft_lowblr"
    "$ROOT/TinyImageNet/checkpoints_fake_ood_ft_moderate"
    "$ROOT/CIFAR-10/checkpoints_abl_A_A1"
    "$ROOT/CIFAR-10/checkpoints_abl_A_A2"
)

for dir in "${stale_dirs[@]}"; do
    remove_path "$dir"
done

find "$ROOT" -type d -name "__pycache__" -print0 |
while IFS= read -r -d '' dir; do
    remove_path "$dir"
done

find "$ROOT/TinyImageNet" -maxdepth 1 -type f \
    \( -name "*.log" -o -name "*.out" \) \
    ! -name "retrain_rich_fake_ood.log" \
    ! -name "tinyimagenet_fake_ood_pipeline_wd005_blr1e3.log" \
    -print0 |
while IFS= read -r -d '' file; do
    remove_path "$file"
done

find "$ROOT/CIFAR-10" -maxdepth 1 -type f \
    \( -name "*.log" -o -name "*.out" -o -name "nohup.out" \) \
    -print0 |
while IFS= read -r -d '' file; do
    remove_path "$file"
done

find "$ROOT/MNIST" -maxdepth 1 -type f \
    \( -name "*.log" -o -name "*.out" \) \
    ! -name "mnist_fake_ood_pipeline.log" \
    -print0 |
while IFS= read -r -d '' file; do
    remove_path "$file"
done

log "Cleanup complete"
