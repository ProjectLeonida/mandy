#!/bin/bash
# Boot gta6rt as its own systemd --user unit, with the memory it used and a digest of the result afterwards.
#
# usage: run-boot.sh <name> <gta6rt.exe> [seconds=150] [KEY=VALUE ...]
#   e.g. run-boot.sh first runtime/build/boot/gta6rt.exe 150 LEONIDA_ARENA_MB=2048
#   LEONIDA_GAME_DIR, LEONIDA_SHADERS and WINE as for boot.sh (from the environment or as KEY=VALUE).
#
# What it adds to boot.sh:
#  - the boot runs outside your shell's process tree, so it survives the terminal;
#  - it waits for memory headroom, then holds a GPU lock (one boot at a time) and a build lock for the whole boot.
#    A build that takes the same lock never competes with the game for memory:
#      flock "${LEONIDA_BOOT_DIR:-$HOME/.local/share/leonida-boot}/.build.lock" cmake --build ... --target gta6rt
#  - RAM, swap, the unit's own memory and VRAM are sampled every 2 s;
#  - a frame capture (every 4800th presented frame) and Wine's error channel are on by default;
#  - after the boot, also when the kernel or systemd-oomd killed it, a digest is written: the unit's result, memory
#    and GPU kernel events, and every distinct error line with its count.
#
# Outputs, in $LEONIDA_BOOT_DIR/runs/<name>/: boot.log  resources.tsv  digest.txt  capture/
# They can quote the title's own log text: keep them on this machine.
set -u
NAME=${1:?name}; EXE=${2:?gta6rt.exe}; SECS=${3:-150}; shift $(( $# >= 3 ? 3 : $# ))
case "$NAME" in *[!A-Za-z0-9_-]*|"") echo "run-boot: the name may hold letters, digits, - and _" >&2; exit 2 ;; esac
[ -f "$EXE" ] || { echo "run-boot: no such gta6rt: $EXE" >&2; exit 2; }
EXE=$(realpath "$EXE")
# gta6rt loads gta6rt_tls_host.dll from its own folder; without it the title gets no module TLS index and exits at once.
[ -f "$(dirname "$EXE")/gta6rt_tls_host.dll" ] || { echo "run-boot: $(dirname "$EXE")/gta6rt_tls_host.dll is missing: build it with the gta6rt target and keep it next to the exe" >&2; exit 2; }
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
OUT=$(realpath -m "${LEONIDA_BOOT_DIR:-$HOME/.local/share/leonida-boot}")
RUN=$OUT/runs/$NAME
UNIT=leonida-boot-$NAME
if systemctl --user is-active --quiet "$UNIT"; then echo "run-boot: $UNIT is already running" >&2; exit 2; fi

declare -A ENV=(
  [WINEPREFIX]="${WINEPREFIX:-$HOME/.local/share/leonida-boot-prefix}"
  [WINEDEBUG]="-all,err+all"
  [GTA6RT]="$EXE"
  [LEONIDA_BOOT_DIR]="$OUT"
  [LEONIDA_CAPTURE]="Z:${RUN//\//\\}\\capture"
  [LEONIDA_CAPTURE_EVERY]=4800
)
for k in WINE LEONIDA_GAME_DIR LEONIDA_SHADERS; do [ -n "${!k:-}" ] && ENV[$k]=${!k}; done
for kv in "$@"; do ENV[${kv%%=*}]="${kv#*=}"; done
[ -n "${ENV[LEONIDA_GAME_DIR]:-}" ] || { echo "run-boot: set LEONIDA_GAME_DIR to the folder of your copy of the package" >&2; exit 2; }
[ -n "${ENV[LEONIDA_SHADERS]:-}" ] || { echo "run-boot: set LEONIDA_SHADERS to the translated shader library" >&2; exit 2; }
args=()
for k in "${!ENV[@]}"; do args+=(--setenv="$k=${ENV[$k]}"); done
mkdir -p "$RUN/capture" || exit 2

echo "run-boot: $UNIT starting (queues for memory headroom, the GPU lock and the build lock)"
rm -f "$RUN/digest.txt" "$RUN/meta"
# OOMScoreAdjust: when the machine runs out of memory, the kernel takes the game and not the session that started it.
systemd-run --user --collect --quiet --unit="$UNIT" -p WorkingDirectory="$OUT" -p OOMScoreAdjust=900 \
  --setenv=BOOT_NAME="$NAME" --setenv=BOOT_SECS="$SECS" "${args[@]}" \
  "$HERE/boot-inner.sh" || { echo "run-boot: could not start $UNIT" >&2; exit 2; }
# The digest is written by a separate, small unit: it survives systemd-oomd killing the boot's cgroup and this shell
# going away. It waits for the boot unit to end (leftover Wine processes included: systemd stops them after the
# title has exited), reads its result from the journal and writes the digest.
systemd-run --user --collect --quiet --unit="leonida-digest-$NAME" -p MemoryMax=256M --setenv=LEONIDA_BOOT_DIR="$OUT" \
  /bin/bash -c 'while s=$(systemctl --user show -p ActiveState --value "$0"); [ "$s" = active ] || [ "$s" = activating ] || [ "$s" = deactivating ]; do sleep 3; done
    START=$(sed -n "s/^start_epoch=//p" "$2/meta" 2>/dev/null)
    RES=$(journalctl --user -u "$0" ${START:+--since "@$START"} -q --no-pager 2>/dev/null | grep -o "Failed with result .[a-z-]*." | tail -1 | tr -d "\047" | sed "s/Failed with result //")
    SERVICE_RESULT=${RES:-success} exec "$3/boot-digest.sh" "$1"' "$UNIT" "$NAME" "$RUN" "$HERE" \
  || { echo "run-boot: could not start the digest unit" >&2; exit 2; }
# Block until the digest exists (so a caller that waits on this script gets the result), then print it.
while [ ! -s "$RUN/digest.txt" ] || systemctl --user is-active --quiet "leonida-digest-$NAME"; do sleep 5; done
echo "run-boot: $UNIT finished. Digest:"
cat "$RUN/digest.txt"
head -1 "$RUN/digest.txt" | grep -q ': OK' && exit 0 || exit 1
