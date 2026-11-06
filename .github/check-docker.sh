#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
engine_image="${1:-pyronear/pyro-engine:test}"
camera_image="${2:-pyronear/pyro-camera-api:test}"

runtime_flags=(
  --read-only --cap-drop=ALL --security-opt=no-new-privileges:true
  --tmpfs /tmp:rw,nosuid,nodev,size=64m,mode=1777
  --tmpfs /usr/src/app/data:rw,nosuid,nodev,size=128m,uid=1000,gid=1000,mode=0700
)

docker run --rm "${runtime_flags[@]}" \
  --mount "type=bind,source=${project_root}/.github/verify_container.py,target=/verify.py,readonly" \
  --mount "type=bind,source=${project_root}/uv.lock,target=/runtime.lock,readonly" \
  "$engine_image" python /verify.py engine

docker run --rm "${runtime_flags[@]}" \
  --tmpfs /usr/src/app/focus_debug:rw,nosuid,nodev,size=128m,uid=1000,gid=1000,mode=0700 \
  --mount "type=bind,source=${project_root}/.github/verify_container.py,target=/verify.py,readonly" \
  --mount "type=bind,source=${project_root}/pyro_camera_api/uv.lock,target=/runtime.lock,readonly" \
  "$camera_image" python /verify.py camera

# The production healthcheck must still stop a stale engine without CAP_KILL or root.
heartbeat_container="$(docker run -d --init "${runtime_flags[@]}" "$engine_image" python -c \
  'import os,time; from pathlib import Path; p=Path("data/heartbeat"); p.touch(); t=time.time()-660; os.utime(p,(t,t)); time.sleep(60)')"
trap 'docker rm -f "$heartbeat_container" >/dev/null 2>&1 || true' EXIT
for _ in {1..30}; do
  if docker exec "$heartbeat_container" test -f data/heartbeat; then break; fi
  sleep 0.2
done
docker exec "$heartbeat_container" sh -c \
  '[ -n "$(find data/heartbeat -mmin +10 2>/dev/null)" ] && kill -9 -1; [ -n "$(find data/heartbeat -mmin -10 2>/dev/null)" ]' || true
exit_code="$(timeout 15 docker wait "$heartbeat_container")"
test "$exit_code" = 137
