#!/bin/sh
# Opens the Leonida builder window (Build.sh is the console version).
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 1
for py in python3.14 python3.13 python3.12 python3; do
    command -v "$py" >/dev/null 2>&1 || continue
    "$py" -c "import sys, tkinter; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >/dev/null 2>&1 && exec "$py" -I "$here/leonida_app.py"
done
echo "Python 3.12+ with tkinter is needed (macOS: brew install python-tk; Debian/Ubuntu: sudo apt install python3-tk)." >&2
exit 1
