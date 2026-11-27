"""Run glslangValidator with a pure-ASCII working directory and pure-ASCII paths.

    python -I runtime/test/glslang_ascii_cwd.py <glslangValidator> [glslang args...]

Windows glslang (16.x, standalone) fast-fails with 0xC0000409 in a non-ASCII working directory, whatever
its arguments say: the CRT resolves files through the narrow (ANSI codepage) current directory, and a
character the codepage cannot represent either mangles the path or crashes the runtime before main. A CMake
build tree under a non-ASCII directory (a user name, a desktop folder) hits this for every shader the test
build compiles. This wrapper keeps every byte glslang sees ASCII: the working directory is the output
directory's 8.3 short form (pure ASCII by construction on the system volume), the output is written there
under its own relative name, and each non-ASCII input file is hard-linked or copied in under a plain ASCII
one - the input's short form cannot be used because 8.3 truncates the extension to three letters (`.comp`
becomes `.com`), and glslang classifies the shader stage from the extension. Off Windows, or when the tree
is already ASCII, this is a plain pass-through.

The pass-through arguments are the validator's own; flags, -D definitions and target environments reach it
untouched. When a path has no 8.3 short form (the volume has short names disabled) the validator runs as it
would without the wrapper - no regression - and a warning says why.
"""
import os
import shutil
import subprocess
import sys


def is_ascii(text):
    return all(ord(c) < 128 for c in text)


if os.name != "nt":
    # The narrow-argv problem is the Windows CRT's; elsewhere exec the validator directly.
    sys.exit(subprocess.run(sys.argv[1:]).returncode)

import ctypes  # noqa: E402  (Windows-only below)


def short_path(path):
    """The 8.3 short form of an existing file or directory, pure ASCII by construction, or None."""
    buf = ctypes.create_unicode_buffer(1024)
    length = ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf))
    return buf.value if 0 < length < len(buf) else None


def main(argv):
    glslang, args = argv[1], list(argv[2:])

    # The -o output first: its directory may not exist yet, and the short form needs it to.
    out_dir, out_name = None, None
    for i, token in enumerate(args[:-1]):
        if token == "-o":
            output = os.path.abspath(args[i + 1])
            out_dir, out_name = os.path.dirname(output), os.path.basename(output)
            os.makedirs(out_dir, exist_ok=True)
            break

    # The working directory decides whether the CRT survives: the output directory's 8.3 short form, or
    # the current directory's own when there is no -o. With no short form (8.3 disabled on the volume)
    # the original directory is used and the validator behaves exactly as without this wrapper.
    work = out_dir or os.getcwd()
    if not is_ascii(work):
        short = short_path(work)
        if short is not None:
            work = short
    usable = is_ascii(work)

    staged = []
    if usable:
        # Relative ASCII names resolve inside the ASCII working directory, so neither the output nor the
        # inputs ever reach glslang with a non-ASCII byte. Inputs keep the extension the stage is
        # classified from; names already ASCII pass through as they are.
        if out_name is not None and not is_ascii(os.path.join(out_dir, out_name)):
            for i, token in enumerate(args[:-1]):
                if token == "-o":
                    args[i + 1] = out_name
                    break
        for i, token in enumerate(args):
            if token.startswith("-") or not os.path.isfile(token):
                continue
            source = os.path.abspath(token)
            if is_ascii(source):
                continue
            extension = os.path.splitext(source)[1]
            if not (is_ascii(extension) and 1 < len(extension) <= 5):
                extension = ".comp"
            index = len(staged)
            while f"_leonida_in_{index}{extension}" in staged:
                index += 1
            name = f"_leonida_in_{index}{extension}"
            target = os.path.join(work, name)
            try:
                if os.path.exists(target):
                    os.remove(target)
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copyfile(source, target)
            except OSError:
                continue
            staged.append(name)
            args[i] = name

    try:
        if not (usable and all(is_ascii(a) for a in args) and is_ascii(glslang)):
            kept = [a for a in [work, glslang, *args] if not is_ascii(a)]
            print(f"glslang_ascii_cwd: no 8.3 short names for {kept!r}; the validator may fail in this tree",
                  file=sys.stderr)
        return subprocess.run([glslang, *args], cwd=work).returncode
    finally:
        for name in staged:
            try:
                os.remove(os.path.join(work, name))
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main(sys.argv))
