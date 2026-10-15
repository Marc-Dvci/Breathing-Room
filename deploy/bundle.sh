#!/usr/bin/env bash
# Package the repository, the models the console loads and the replayed recording, and
# upload the bundle for setup_instance.sh. Run from the repository root.
#   AWS_PROFILE=<profile> bash deploy/bundle.sh
set -euo pipefail
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
DATA=${BR_DATA:-../data}
STAGE=$(mktemp -d)/breathing-room
mkdir -p "$STAGE/models" "$STAGE/data/fzj/videos"
git ls-files --cached --others --exclude-standard | tar -cf - -T - | tar -xf - -C "$STAGE"
cp models/aliked-n16rot-top1k-640.onnx models/aliked_lightglue.onnx models/dmcount_sha.onnx models/dmcount_qnrf.onnx "$STAGE/models/"
cp "$DATA/fzj/videos/150_q_56_h0.mp4" "$STAGE/data/fzj/videos/"
tar -czf "$STAGE/../bundle.tgz" -C "$STAGE/.." breathing-room
aws s3 cp "$STAGE/../bundle.tgz" "s3://breathingroom-deploy-$ACCOUNT/bundle.tgz" --only-show-errors
ls -la "$STAGE/../bundle.tgz"
