#!/usr/bin/env bash
# Rebuild the bundle and re-run setup_instance.sh on the instance (idempotent, keeps var/).
#   AWS_PROFILE=<profile> bash deploy/redeploy.sh <instance-id> <host>
set -euo pipefail
export MSYS_NO_PATHCONV=1
IID=${1:?instance id}; HOST=${2:?host}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
bash deploy/bundle.sh >/dev/null
aws ssm send-command --instance-ids "$IID" --document-name AWS-RunShellScript --query Command.CommandId --output text \
  --parameters "{\"executionTimeout\":[\"3600\"],\"commands\":[\"cd /var/tmp && rm -rf breathing-room && aws s3 cp s3://breathingroom-deploy-$ACCOUNT/bundle.tgz . --only-show-errors && tar -xzf bundle.tgz && rm bundle.tgz && bash breathing-room/deploy/setup_instance.sh $HOST 2>&1 | tail -5\"]}"
