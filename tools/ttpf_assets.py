"""Complete the rpf9dump texture-pool manifest during the assets build."""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import NamedTuple
import zlib

# `python -I` implies `-P`, so the script's directory is not on sys.path; the sibling modules are next to it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bcpack
from file_mode import published_mode
import ttpf_mips as ttpf


# The modules that decode a full chain, next to this file. Both pass stamps hash them, so a pool completed by an
# older decoder is done again, as one completed with another calibration is.
DECODER_SOURCES = ("bcpack.py", "bcpack_tables.py", "bcpack_tables_bptc.py", "calibrated_texture.py",
                   "tex_headers.py", "ttpf_mips.py")


def decoder_digest():
    h = hashlib.sha256()
    for name in DECODER_SOURCES:
        data = Path(__file__).with_name(name).read_bytes()   # a missing source is an error, not an empty hash
        h.update(f"{name}\0{len(data)}\0".encode())
        h.update(data)
    return h.hexdigest()


def strict_stamp(raw_history=None):
    # Ends in "-strict" (candidate() keeps such rows without a profile); its first 19 characters name the DDS.
    digest = decoder_digest()
    if raw_history is not None:
        digest = hashlib.sha256(digest.encode() + raw_history).hexdigest()
    return f"v1-{digest[:16]}-strict"


class Profiles(NamedTuple):
    calibration: dict | None = None
    history: dict | None = None
    strict_stamp: str | None = None


def load_history(work):
    """Read a combined history once and check its tables before opening any pool."""
    path = Path(work, "bcpack-history.json")
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        raise ValueError(f"{path}: cannot read the history: {exc.strerror or exc}") from None
    try:
        history = json.loads(raw)
        keys = {name.lower(): codec for codec, name in bcpack.CODEC_NAMES.items()}
        if not isinstance(history, dict) or not history or history.keys() - keys.keys():
            raise ValueError("expected a combined BCPack history object keyed by codec")
        for key, section in history.items():
            if not isinstance(section, dict):
                raise ValueError(f"{key} history section must be a JSON object")
        # The checks of bcpack_history_calib.py --verify: the table loaders check shapes only, so a confirmed depth
        # above the known one, or without values, would publish a provisional decode as a strict DDS. Imported here,
        # with a history only: the module reconfigures stdout and stderr when it loads.
        import bcpack_history_calib
        bcpack_history_calib.check_history_tables(history, [keys[key] for key in history])
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError(f"{path}: invalid BCPack history: {exc}") from None
    return history, strict_stamp(raw)


def load_profile(work):
    history, strict = load_history(work)
    path = Path(work, "bcpack-calibration.json")
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return strict or strict_stamp(), Profiles(history=history, strict_stamp=strict)
    except OSError as exc:
        raise ValueError(f"{path}: cannot read the calibration profile: {exc.strerror or exc}") from None
    try:
        # json.loads() on the bytes also detects a UTF-16 profile or one saved with a BOM.
        profile = json.loads(raw)
    except ValueError as exc:   # JSONDecodeError, or an encoding json cannot detect
        # The profile is hand-written (make_game says where it goes), so a typo names the file, not a bare
        # 'Expecting property name enclosed in double quotes'.
        raise ValueError(f"{path}: not a JSON calibration profile: {exc}") from None
    if not isinstance(profile, dict) or profile.get("version") != 1:
        raise ValueError(f"Unsupported calibration version in {path}")
    try:
        from calibrated_texture import native
        native()
    except (OSError, ImportError) as exc:
        raise RuntimeError("Calibrated texture export requires NumPy and the release build of "
                           "tools/bcpack_calibration (see its README.md)") from exc
    stamp = "v1-" + hashlib.sha256(decoder_digest().encode() + raw).hexdigest()
    if strict is not None:
        stamp = f"{stamp}+{strict}"
    return stamp, Profiles(profile, history, strict)


def publish(path, data):
    f = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    tmp = Path(f.name)
    try:
        with f:
            f.write(data)
        if os.name != "nt":
            tmp.chmod(published_mode(path))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def file_identity(path):
    """(device, inode) of an existing file, or None: two names with one identity are one file on disk."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino) if st.st_ino else None


def remove_replaced(directory, replaced, rows):
    """Remove the files of `replaced` (names the rows had before they were switched) from `directory`; returns
    the names that could not be removed (in use), sorted.

    Only a .dds file by its bare name in this folder is removed, never manifest.json, and never a file a row still
    names: by the same name, by a name that differs in letter case only, or by any other spelling of the same file
    on disk (on Windows and macOS that can be the file the pass has just published). Each file is looked at once:
    one stat per named row and one per replaced name, not one per pair."""
    names = [row["File"] for row in rows if isinstance(row.get("File"), str)]
    named = {os.path.normcase(name) for name in names}
    on_disk = {file_identity(directory / name) for name in names} - {None}
    left = []
    for name in sorted(replaced):
        if Path(name).name != name or not name.lower().endswith(".dds") or os.path.normcase(name) in named:
            continue
        if file_identity(directory / name) in on_disk:
            continue
        try:
            (directory / name).unlink(missing_ok=True)
        except OSError:
            left.append(name)
    return left


# The manifest key that lists replaced files still to be removed. It is written with the manifest that switches the
# rows, so a removal that fails (a file in use) or a run stopped right after the switch is finished by a later run,
# even though that run finds the pass stamped and writes nothing else.
PENDING = "BCPackReplaced"


def finish_replaced(directory, manifest):
    """Remove the files the manifest lists under PENDING, and publish it again with what is left."""
    pending = manifest.get(PENDING)
    if not isinstance(pending, list) or not pending:
        return
    left = remove_replaced(directory, {name for name in pending if isinstance(name, str)}, manifest["Textures"])
    if left:
        manifest[PENDING] = left
    else:
        manifest.pop(PENDING)
    if left != pending:
        try:
            publish(directory / "manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
        except OSError as exc:
            # The files are gone; the list on disk still names them, and a later run drops them (missing is fine).
            print(f"assets: {directory.name}: could not update manifest.json after removing replaced DDS files: "
                  f"{exc.strerror or exc}", file=sys.stderr)
    if left:
        print(f"assets: {directory.name}: {len(left)} replaced DDS file(s) could not be removed (in use?) and stay "
              "next to the full-chain files; the next build tries again", file=sys.stderr)


def candidate(row, profiles):
    if row.get("DxgiFormat") not in ttpf.BC_FORMATS:
        return False
    if row.get("BCPackDecoder"):
        return profiles.calibration is not None or row["BCPackDecoder"].endswith("-strict")
    return (row["File"] is None or row.get("BaseMipLevel", 0) > 0
            or row.get("Mips", 0) < row.get("SourceMips", 0))


def complete_pool(source, directory, manifest, settings):
    """Publish full chains next to ordinary DDS files, then switch the catalog to them."""
    stamp, profiles = settings
    directory = Path(directory)
    if manifest.get("BCPackPass") == stamp:
        finish_replaced(directory, manifest)
        return 0, 0
    rows = [row for row in manifest["Textures"] if candidate(row, profiles)]
    if not rows:
        return 0, 0
    written = blocked = 0
    superseded = set()   # the files the rows named before they were switched to a full chain
    with closing(ttpf.Pool(source)) as pool:
        decoder = None
        strict = profiles.strict_stamp
        profile = profiles.calibration
        if profile is not None:
            from calibrated_texture import LocalDecoder
            decoder = LocalDecoder(pool, profile)
        try:
            for row in rows:
                index = row["Index"]
                desc = pool.descriptor(index)
                base = desc["base"]
                if not pool.chunks or base < pool.chunks[0][0] or base >= pool.virtual_size:
                    continue
                span = pool.span(index, base)
                if not span:
                    continue
                try:
                    if not pool.codecs(base, span) & bcpack.SUPPORTED_REBUILD_CODECS:
                        continue
                    if decoder and desc["dimension"] == 1 and desc["format"] in decoder.formats:
                        method = stamp.partition("+")[0]
                        blob = decoder.dds(index)
                    else:
                        if profile is not None and strict is None:
                            strict = strict_stamp()   # once per pass: it reads and hashes six sources
                        method = strict or stamp
                        analysis = ttpf.analyze(pool, index)
                        blob = ttpf.texture_dds(pool, analysis, history=profiles.history)
                except (ValueError, KeyError, IndexError, ttpf.BcpackRequired,
                        bcpack.BcpackUnsupportedCodec, zlib.error) as exc:
                    row["BCPackLimitation"] = str(exc)
                    blocked += 1
                    continue
                # A distinct name keeps an existing lower-mip DDS usable until the manifest switches.
                name = f"{index:06d}_bcpack_{method[:19]}.dds"
                publish(directory / name, blob)
                if isinstance(row.get("File"), str) and row["File"] != name:
                    superseded.add(row["File"])
                row.update(File=name, Mips=desc["mips"], BaseMipLevel=0, Limitation=None, BCPackDecoder=method)
                row.pop("BCPackLimitation", None)
                written += 1
        finally:
            if decoder:
                decoder.read_chunk.cache_clear()
    manifest["BCPackPass"] = stamp
    # The files the switched rows named before would otherwise stay in the folder, named by no row, so every
    # completed texture would be stored twice. They are listed in the manifest that switches the rows and removed
    # only after it is published: a file another row still names is kept, and one that can't be removed (in use)
    # stays listed for the next build. The export is complete either way.
    pending = {name for name in manifest.get(PENDING) or [] if isinstance(name, str)} | superseded
    if pending:
        manifest[PENDING] = sorted(pending)
    publish(directory / "manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
    finish_replaced(directory, manifest)
    return written, blocked
