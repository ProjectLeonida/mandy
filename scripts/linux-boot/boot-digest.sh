#!/bin/bash
# usage: boot-digest.sh <name>      (run-boot.sh starts it when the leonida-boot-<name> unit has ended)
# Runs after the boot however it ended (exit, crash, oomd kill, time limit). Writes
# $LEONIDA_BOOT_DIR/runs/<name>/digest.txt and appends one line to $LEONIDA_BOOT_DIR/boots.tsv.
# Local only: a digest quotes trace lines, so it never goes into a commit, an issue or a pull request.
set -u
NAME=${1:?name}; BASE=${LEONIDA_BOOT_DIR:-$HOME/.local/share/leonida-boot}; RUN=$BASE/runs/$NAME
LOG=$RUN/boot.log; RES=$RUN/resources.tsv; META=$RUN/meta; OUT=$RUN/digest.txt; LEDGER=$BASE/boots.tsv
START=$(sed -n 's/^start_epoch=//p' "$META" 2>/dev/null); BOOT=$(sed -n 's/^boot_epoch=//p' "$META" 2>/dev/null)
START=${START:-$(date +%s)}; BOOT=${BOOT:-$START}; END=$(date +%s)
RESULT=${SERVICE_RESULT:-?}
D3D=$(grep -o 'd3d12=[^ ]*' "$LOG" 2>/dev/null | tail -1 | cut -d= -f2)
STDOUT=${D3D/d3d12-run-/stdout-run-}
CAP=$RUN/capture

verdict=OK
# systemd-oomd can kill the unit's processes without the unit ending in "oom-kill" (boot.sh then exits on its own).
if journalctl --user -u "leonida-boot-$NAME" --since "@$START" -q --no-pager 2>/dev/null | grep -q 'systemd-oomd killed'; then RESULT=oom-kill; fi
# The loader's own exit code, as boot-inner.sh logged it (124 is the time limit and was turned into 0 there).
RC=$(sed -n 's/^\[run-boot\] .* boot\.sh exit \([0-9]*\)$/\1/p' "$LOG" 2>/dev/null | tail -1)
[ "$RESULT" = success ] && [ -n "$RC" ] && [ "$RC" != 0 ] && [ "$RC" != 124 ] && RESULT=exit-code
case "$RESULT" in
  oom-kill) verdict="KILLED BY OOM (systemd-oomd or kernel): memory, not a crash" ;;
  timeout) verdict="TIMEOUT" ;;
  exit-code) verdict="EXITED WITH ERROR (boot.sh exit ${RC:-?})" ;;
  signal|core-dump) verdict="KILLED BY SIGNAL" ;;
  success) verdict="OK (ran to its time limit or exited cleanly)" ;;
esac
grep -q 'time limit reached' "$LOG" 2>/dev/null && [ "$RESULT" = success ] && verdict="OK (ran its full time limit)"

cnt() { local c=0; [ -f "$D3D" ] && c=$(grep -a -c -E "$@" "$D3D"); echo "${c:-0}"; }
{
  echo "== leonida boot $NAME: $verdict"
  echo "unit result=$RESULT"
  echo "queued $((BOOT-START)) s, ran $((END-BOOT)) s; trace $D3D"
  sed -n '/^LEONIDA_\|^WINEDEBUG\|^unit=/p' "$META" 2>/dev/null | tr '\n' ' '; echo
  echo
  echo "-- counts in the D3D12 trace"
  echo "no translated shader $(cnt 'no translated ')  E_NOTIMPL $(cnt 'E_NOTIMPL')  failed with an HRESULT $(cnt 'failed with 0x8')"
  echo "Wine err: lines $(cnt ':err:')  exceptions $(cnt -i 'exception|unhandled|page fault|access violation')"
  echo
  echo "-- frames (every LEONIDA_CAPTURE_EVERY-th presented frame of the swapchain)"
  if [ -f "$CAP/index.txt" ]; then
    echo "$(wc -l < "$CAP/index.txt") captured; last: $(tail -1 "$CAP/index.txt" | cut -c1-110)"
    awk '{m=$(NF-2)+$(NF-1)+$NF; if (m>0) n++} END{print n+0 " of " NR " frames non-black"}' "$CAP/index.txt"
  else
    echo "none captured"
  fi
  echo
  echo "-- resources (2 s samples)"
  [ -s "$RES" ] && awk -F'\t' 'NR>1{ if(min==""||$2<min)min=$2; if($3>sw)sw=$3; if($4>um)um=$4; if($5>us)us=$5; if($6+0>gm)gm=$6 } END{printf "min RAM available %d MB, max swap used %d MB, game peak %d MB RAM + %d MB swap, VRAM peak %d MB\n", min, sw, um, us, gm}' "$RES"
  journalctl --user -u "leonida-boot-$NAME" --since "@$START" -q --no-pager 2>/dev/null | grep -E 'Consumed|oom|Failed' | sed 's/^.*\]: //' | sort -u | cut -c1-200
  echo
  echo "-- system events during the boot (OOM, GPU Xid, segfaults)"
  { journalctl --since "@$BOOT" -q --no-pager 2>/dev/null | grep -i -E 'systemd-oomd.*(Marked|killed)|oom-kill|out of memory|NVRM: Xid|segfault|gta6rt|wineserver.*crash' ; } | cut -c1-200 | tail -15
  echo
  echo "-- distinct error lines (normalised, with counts; top 60)"
  for f in "$D3D" "$STDOUT"; do [ -f "$f" ] && cat "$f"; done 2>/dev/null \
    | grep -a -i -E 'error|fail|E_NOTIMPL|E_INVALIDARG|E_FAIL|0x8[0-9a-f]{7}|exception|assert|abort|unhandled|page fault|refused|not implemented|no translated|no emulation|crash|:err:|fatal|cannot|could not|unsupported|invalid' \
    | grep -a -v -E ' ok$| called$' \
    | sed -E 's/#[0-9]+/#N/g; s/0x[0-9a-fA-F]{6,}/0xADDR/g; s/\b[0-9a-f]{64}\b/SHA/g; s/[0-9]{4,}/NUM/g; s/^[0-9a-f]{4}:/TID:/' \
    | cut -c1-220 | sort | uniq -c | sort -rn | head -60
  echo
  echo "-- where the trace ended"
  [ -f "$D3D" ] && tail -n 8 "$D3D" | cut -c1-200
} > "$OUT.tmp" 2>&1
mv "$OUT.tmp" "$OUT"

frames=$( [ -f "$CAP/index.txt" ] && awk '{m=$(NF-2)+$(NF-1)+$NF; if (m>0) n++} END{print (n+0) "/" NR}' "$CAP/index.txt" || echo "-")
peak=$( [ -s "$RES" ] && awk -F'\t' 'NR>1{if($4>u)u=$4} END{print u+0}' "$RES" || echo "-")
exec 7>>"$LEDGER"; flock 7
[ -s "$LEDGER" ] || printf 'end\tname\tresult\tran_s\tnonblack_frames\tpeak_mb\tgta6rt\n' >&7
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$NAME" "$RESULT" "$((END-BOOT))" "$frames" "$peak" \
  "$(sed -n 's/^unit=.*gta6rt=\([^ ]*\).*/\1/p' "$META")" >&7
exit 0
