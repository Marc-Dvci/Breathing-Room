#!/usr/bin/env bash
# Runs ON the Graviton instance (Amazon Linux 2023, arm64), as root, through SSM:
#
#   aws ssm send-command --document-name AWS-RunShellScript --instance-ids <id> \
#     --parameters 'commands=["aws s3 cp s3://<bucket>/bundle.tgz /tmp/ && tar -xzf /tmp/bundle.tgz -C /tmp && bash /tmp/breathing-room/deploy/setup_instance.sh <host>"]'
#
# Installs Python 3.11, the pinned dependencies with the headless OpenCV 5 aarch64 wheel,
# JuPedSim built from its sdist, Caddy for TLS, and the systemd service. Idempotent.
set -euo pipefail
HOST=${1:?usage: setup_instance.sh <host, e.g. 1-2-3-4.sslip.io>}
SRC=$(cd "$(dirname "$0")/.." && pwd)
export TMPDIR=/var/tmp  # /tmp is a RAM-backed tmpfs on Amazon Linux 2023: too small for pip builds
APP=/opt/breathingroom
CADDY_VERSION=2.10.2
JUPEDSIM_VERSION=1.4.2

dnf install -y -q python3.11 python3.11-devel gcc-c++ cmake git tar gzip >/dev/null
if [ ! -f /swapfile ]; then  # t4g.small has 2 GB of RAM: 2 GB of swap covers the model loads
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  echo "/swapfile none swap sw 0 0" >> /etc/fstab
fi
id breathingroom >/dev/null 2>&1 || useradd --system --home-dir $APP --shell /sbin/nologin breathingroom

mkdir -p $APP
rsync -a --delete --exclude var/ --exclude .venv/ $SRC/ $APP/ 2>/dev/null || { dnf install -y -q rsync; rsync -a --delete --exclude var/ --exclude .venv/ $SRC/ $APP/; }
mkdir -p $APP/var
chown -R breathingroom:breathingroom $APP

if [ ! -x $APP/.venv/bin/python ]; then
  python3.11 -m venv $APP/.venv
fi
$APP/.venv/bin/pip install -q --upgrade pip
$APP/.venv/bin/pip install -q -r $APP/deploy/requirements-graviton.txt
if ! $APP/.venv/bin/python -c "import jupedsim" 2>/dev/null; then
  # no aarch64 Linux wheel on PyPI: build the C++ core from the sdist, without the GUI extras
  $APP/.venv/bin/pip install -q --no-deps --no-binary jupedsim "jupedsim==$JUPEDSIM_VERSION"
fi
$APP/.venv/bin/python -c "import cv2, jupedsim; print('OpenCV', cv2.__version__, '| JuPedSim', jupedsim.__version__)"

if [ ! -x /usr/local/bin/caddy ]; then
  curl -sSL -o /tmp/caddy.tgz "https://github.com/caddyserver/caddy/releases/download/v$CADDY_VERSION/caddy_${CADDY_VERSION}_linux_arm64.tar.gz"
  tar -xzf /tmp/caddy.tgz -C /usr/local/bin caddy
  id caddy >/dev/null 2>&1 || useradd --system --home-dir /var/lib/caddy --create-home --shell /sbin/nologin caddy
fi
mkdir -p /etc/caddy
sed "s/{\$BR_HOST}/$HOST/" $APP/deploy/Caddyfile > /etc/caddy/Caddyfile
install -m 644 $APP/deploy/caddy.service /etc/systemd/system/caddy.service

TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
REGION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region)
cat > /etc/breathingroom.env <<EOF
BR_DATA=$APP/data
BR_TRACE=$APP/var/trace.jsonl
BR_USE_MODEL=1
BR_IDLE_PAUSE_S=120
BR_ISOLATE_MODELS=1
BR_SCREEN_EVERY_S=3
BR_CONFIRM_EVERY_S=600
BR_CLOUDWATCH_GROUP=/breathingroom/trace
BR_CLOUDWATCH_STREAM=console
AWS_REGION=$REGION
AWS_DEFAULT_REGION=$REGION
EOF
install -m 644 $APP/deploy/breathingroom.service /etc/systemd/system/breathingroom.service
systemctl daemon-reload
systemctl enable --now breathingroom caddy
systemctl restart breathingroom caddy
sleep 8
curl -s http://127.0.0.1:8000/health && echo
