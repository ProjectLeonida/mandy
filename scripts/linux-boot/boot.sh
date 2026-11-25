#!/bin/bash
# Boot the loader under Wine or Proton with the network off (unshare -rn: a new user and network namespace, no root).
# usage: boot.sh dry-run|init|run [seconds]     (run stops the title after `seconds`, default 150)
#   LEONIDA_GAME_DIR  the folder of your copy of the package (the one with game_scarlett_master.exe); required
#   LEONIDA_SHADERS   the translated shader library (<work>/shaders/spirv of make_game.py); required for run
#   LEONIDA_BOOT_DIR  where traces, the overlay and the driver's shader cache go (default ~/.local/share/leonida-boot)
#   GTA6RT            the loader to boot (default: runtime/build/boot/gta6rt.exe of this checkout)
#   WINE, WINEPREFIX  the Wine to use (default: Proton Experimental's files/bin/wine when Steam has it, else `wine`)
#                     and a prefix kept for boots alone. Use one Wine per prefix.
# Traces stay on this machine: they can quote game data, so never post them.
set -u
MODE=${1:?dry-run|init|run}; SECS=${2:-150}
case "$MODE" in dry-run|init|run) ;; *) echo "boot.sh: the mode is dry-run, init or run" >&2; exit 2 ;; esac
[ -n "${LEONIDA_GAME_DIR:-}" ] || { echo "boot.sh: set LEONIDA_GAME_DIR to the folder of your copy of the package" >&2; exit 2; }
GAME=$(realpath -m "$LEONIDA_GAME_DIR")
EXE=$GAME/game_scarlett_master.exe
[ -f "$EXE" ] || { echo "boot.sh: $EXE not found" >&2; exit 2; }
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)   # the checkout this script is in
OUT=$(realpath -m "${LEONIDA_BOOT_DIR:-$HOME/.local/share/leonida-boot}")
GTA6RT=${GTA6RT:-$REPO/runtime/build/boot/gta6rt.exe}
[ -f "$GTA6RT" ] || { echo "boot.sh: no loader at $GTA6RT (build the gta6rt target or set GTA6RT)" >&2; exit 2; }
if [ "$MODE" = run ]; then
  [ -n "${LEONIDA_SHADERS:-}" ] || { echo "boot.sh: set LEONIDA_SHADERS to the translated shader library" >&2; exit 2; }
  SHADERS=$(realpath -m "$LEONIDA_SHADERS")
  [ -d "$SHADERS" ] || { echo "boot.sh: no shader library at $SHADERS" >&2; exit 2; }
  export LEONIDA_SHADERS="$SHADERS" LEONIDA_D3D12=vulkan
fi
STAMP=$(date +%H%M%S)
T=$OUT/traces; mkdir -p "$T" "$OUT/overlay" "$OUT/cache/nv" || exit 2

win() { local p=${1//\//\\}; printf 'Z:%s' "$p"; }   # a Linux path as Wine's Z: drive sees it
PROTON="$HOME/.local/share/Steam/steamapps/common/Proton - Experimental/files/bin/wine"
WINE=${WINE:-$( [ -x "$PROTON" ] && echo "$PROTON" || echo wine )}
export WINEPREFIX=${WINEPREFIX:-$HOME/.local/share/leonida-boot-prefix} WINEDEBUG=${WINEDEBUG:--all}
# No Mono or Gecko install prompt in a new prefix, and no crash debugger: a process that crashes ends, instead of
# starting winedbg (which can crash in turn and start itself again without end).
export WINEDLLOVERRIDES="mscoree,mshtml=;winedbg.exe=d"
# NVIDIA's shader disk cache: kept under the boot folder, with a limit large enough to keep a library's pipelines.
export __GL_SHADER_DISK_CACHE=1 __GL_SHADER_DISK_CACHE_PATH=$OUT/cache/nv
export __GL_SHADER_DISK_CACHE_SIZE=${__GL_SHADER_DISK_CACHE_SIZE:-21474836480} __GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1

TRACE=$T/boot-$MODE-$STAMP.jsonl
D3D=$T/d3d12-$MODE-$STAMP.log
args=(--"$MODE" "$(win "$EXE")" --game-dir "$(win "$GAME")" --overlay "$(win "$OUT/overlay")" --trace "$(win "$TRACE")")
[ "$MODE" = run ] && args+=(--trace-d3d12)
echo "[boot] $MODE  gta6rt=$GTA6RT  trace=$TRACE  d3d12=$D3D"
if [ "$MODE" = run ]; then
  timeout --signal=INT --kill-after=20 "$SECS" unshare -rn "$WINE" "$GTA6RT" "${args[@]}" > "$T/stdout-$MODE-$STAMP.log" 2> "$D3D"
else
  unshare -rn "$WINE" "$GTA6RT" "${args[@]}" > "$T/stdout-$MODE-$STAMP.log" 2> "$D3D"
fi
rc=$?; echo "[boot] exit=$rc"
summary=("$TRACE"); [ "$MODE" = run ] && summary+=(--d3d12 "$D3D")
python3 -I "$REPO/tools/boot_trace.py" "${summary[@]}" > "$T/summary-$MODE-$STAMP.txt" 2>&1
echo "[boot] summary: $T/summary-$MODE-$STAMP.txt"
exit $rc
