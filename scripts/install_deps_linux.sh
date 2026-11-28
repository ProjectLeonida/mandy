#!/usr/bin/env bash
# Installs the Linux prerequisites of docs/linux.md ("Build the port") with the distribution's package manager:
# Python 3.12+, llvm-mc 21+ with the AMDGPU target, SPIRV-Tools, CMake, Ninja, a C++ compiler and the Vulkan
# headers and loader, plus the optional ones (.NET SDK, ffmpeg, zenity, Tk) where the distribution packages them.
#
#   scripts/install_deps_linux.sh             show what it will do, ask, then install
#   scripts/install_deps_linux.sh --dry-run   only show the commands
#   scripts/install_deps_linux.sh --yes       don't ask
#   scripts/install_deps_linux.sh --minimal   leave out .NET, ffmpeg, zenity and Tk
#
# Supported: apt (Debian, Ubuntu and derivatives such as Parrot or Kali), dnf (Fedora), pacman (Arch). On apt it takes
# the newest llvm-21 or later the system's own sources have (in November 2026: llvm-22 on Debian 12 and 13, llvm-21 on
# Ubuntu 22.04 and 24.04, llvm-22 on Ubuntu 26.04), so no extra repository is needed. When the install is done it
# checks what the build needs, Python 3.12+ included (Debian 12 and Ubuntu 22.04 ship 3.11 and 3.10).
#
# Nothing of the game is downloaded or touched; packages come from the distribution only.
set -euo pipefail

LLVM_MIN=21
LLVM_TRY=(23 22 21)

dry_run=0
assume_yes=0
minimal=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=1 ;;
        --yes|-y) assume_yes=1 ;;
        --minimal) minimal=1 ;;
        -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $arg (see --help)" >&2; exit 2 ;;
    esac
done

say() { printf '%s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

if [ "$(id -u)" -eq 0 ]; then
    SUDO=()
elif command -v sudo >/dev/null 2>&1; then
    SUDO=(sudo)
else
    die "run this as root or install sudo"
fi

SU=${SUDO[*]:+${SUDO[*]} }

# Commands are collected first, shown, and run only after the user agrees.
PLAN=()
plan() { PLAN+=("$*"); }

[ -r /etc/os-release ] || die "/etc/os-release is missing; see docs/linux.md for the package list"
# shellcheck disable=SC1091
. /etc/os-release
ID=${ID:-unknown}
ID_LIKE=${ID_LIKE:-}
is_like() { case " $ID $ID_LIKE " in *" $1 "*) return 0 ;; esac; return 1; }

confirm() {
    local answer
    [ "$assume_yes" -eq 1 ] && return 0
    read -r -p "$1 [y/N] " answer || return 1
    case "$answer" in y|Y|yes|o|O|oui) return 0 ;; *) return 1 ;; esac
}

apt_has() { apt-cache show "$1" >/dev/null 2>&1; }

setup_apt() {
    local llvm="" v
    # Package lists that were never fetched (a fresh container) would make every package look missing.
    if ! apt_has git; then
        if [ "$dry_run" -eq 1 ]; then
            die "apt has no package lists yet; run '${SU}apt-get update' first"
        fi
        say "Package lists are needed to choose an LLVM version: ${SU}apt-get update"
        confirm "Refresh the package lists now?" || { say "Nothing installed."; exit 1; }
        say "+ ${SU}apt-get update"
        "${SUDO[@]}" apt-get update
    else
        plan "${SU}apt-get update"
    fi
    for v in "${LLVM_TRY[@]}"; do
        if apt_has "llvm-$v"; then llvm=$v; break; fi
    done
    [ -n "$llvm" ] || die "no llvm-$LLVM_MIN or newer in your apt sources; add LLVM's own repository (https://apt.llvm.org) and run this again"
    local pkgs=(git python3 "llvm-$llvm" spirv-tools cmake ninja-build g++ libvulkan-dev)
    local optional=()
    if [ "$minimal" -eq 0 ]; then
        for p in ffmpeg zenity python3-tk; do
            if apt_has "$p"; then optional+=("$p"); else warn "$p isn't packaged here; skipped (optional)"; fi
        done
        if apt_has dotnet-sdk-10.0; then optional+=(dotnet-sdk-10.0)
        elif apt_has dotnet-sdk-8.0; then optional+=(dotnet-sdk-8.0)
        else warn ".NET SDK isn't packaged here; the assets stage needs it: https://learn.microsoft.com/dotnet/core/install/linux"
        fi
    fi
    plan "${SU}apt-get install -y ${pkgs[*]} ${optional[*]}"
}

setup_dnf() {
    local pkgs=(git python3 llvm spirv-tools cmake ninja-build gcc-c++ vulkan-headers vulkan-loader-devel)
    local optional=()
    if [ "$minimal" -eq 0 ]; then
        for p in ffmpeg-free zenity python3-tkinter; do
            if dnf -q info "$p" >/dev/null 2>&1; then optional+=("$p"); else warn "$p isn't available; skipped (optional)"; fi
        done
        if dnf -q info dotnet-sdk-10.0 >/dev/null 2>&1; then optional+=(dotnet-sdk-10.0)
        elif dnf -q info dotnet-sdk-8.0 >/dev/null 2>&1; then optional+=(dotnet-sdk-8.0)
        else warn ".NET SDK isn't available; the assets stage needs it"
        fi
    fi
    plan "${SU}dnf install -y ${pkgs[*]} ${optional[*]}"
}

setup_pacman() {
    local pkgs=(git python llvm spirv-tools cmake ninja gcc vulkan-headers vulkan-icd-loader)
    [ "$minimal" -eq 0 ] && pkgs+=(dotnet-sdk ffmpeg zenity tk)
    plan "${SU}pacman -S --needed --noconfirm ${pkgs[*]}"
}

if command -v apt-get >/dev/null 2>&1 && { is_like debian || is_like ubuntu; }; then
    setup_apt
elif command -v dnf >/dev/null 2>&1; then
    setup_dnf
elif command -v pacman >/dev/null 2>&1; then
    setup_pacman
else
    die "no supported package manager (apt, dnf, pacman) on '$ID'; docs/linux.md lists what to install"
fi

say ""
say "Detected: ${PRETTY_NAME:-$ID}"
say "Commands:"
for c in "${PLAN[@]}"; do say "  $c"; done
say ""
[ "$dry_run" -eq 1 ] && exit 0

confirm "Run them?" || { say "Nothing installed."; exit 1; }
for c in "${PLAN[@]}"; do
    say "+ $c"
    bash -c "$c"
done

# What the build checks for, checked once more after the install.
say ""
say "Check:"
status=0
check() { if "$@" >/dev/null 2>&1; then say "  ok       $label"; else say "  MISSING  $label"; status=1; fi; }
label="Python 3.12+"; check python3 -c 'import sys; sys.exit(sys.version_info < (3, 12))'
mc=""
for c in llvm-mc llvm-mc-23 llvm-mc-22 llvm-mc-21; do
    command -v "$c" >/dev/null 2>&1 || continue
    ver=$("$c" --version 2>/dev/null | sed -n 's/.*LLVM version \([0-9]*\).*/\1/p' | head -1)
    if [ -n "$ver" ] && [ "$ver" -ge "$LLVM_MIN" ] && "$c" --version 2>/dev/null | grep -q amdgcn; then mc=$c; break; fi
done
label="llvm-mc $LLVM_MIN+ with AMDGPU${mc:+ ($mc)}"; check test -n "$mc"
label="spirv-as, spirv-val"; check sh -c 'command -v spirv-as && command -v spirv-val'
label="CMake 3.24+"; check python3 -c 'import re, subprocess, sys; v = re.search(r"(\d+)\.(\d+)", subprocess.run(["cmake", "--version"], capture_output=True, text=True).stdout); sys.exit(not v or (int(v[1]), int(v[2])) < (3, 24))'
label="Ninja"; check command -v ninja
label="C++ compiler"; check sh -c 'command -v g++ || command -v clang++'
label="Vulkan headers"; check test -e /usr/include/vulkan/vulkan.h
if [ "$minimal" -eq 0 ]; then
    if command -v dotnet >/dev/null 2>&1; then say "  ok       .NET SDK (optional)"; else say "  -        .NET SDK (optional: the assets stage)"; fi
fi
say ""
if [ "$status" -eq 0 ]; then
    say "Done. Next: ./Build.sh --check with your own copy of the game (docs/linux.md)."
else
    say "Something is still missing (above); docs/linux.md explains each one."
fi
exit "$status"
