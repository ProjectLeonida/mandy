"""DirectStorage request vectors from independent synthetic GDK goldens."""
from pathlib import Path
import sys
import zlib


def write(out):
    source = Path(__file__).resolve().parents[2] / "tests/fixtures/bcpack-standalone.txt"
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        codec, name, packed, expected = line.split()
        if not name.endswith("-request"):
            continue
        raw, plain = bytes.fromhex(packed), bytes.fromhex(expected)
        assert len(raw) < len(plain) and len(zlib.compress(raw)) < len(plain)
        for label, data in (("raw", raw), ("plain", plain), ("wrapped", zlib.compress(raw))):
            print(f"static constexpr uint8_t {label}{codec}[] = {{", file=out)
            for i in range(0, len(data), 24):
                print("    " + ", ".join(map(str, data[i:i+24])) + ",", file=out)
            print("};", file=out)
    print("static constexpr Vector vectors[] = {", file=out)
    for codec in range(1, 8):
        print(f"    {{{codec}, raw{codec}, plain{codec}, wrapped{codec}}},", file=out)
    print("};", file=out)


if __name__ == "__main__":
    with Path(sys.argv[1]).open("w", encoding="utf-8", newline="\n") as output:
        write(output)
