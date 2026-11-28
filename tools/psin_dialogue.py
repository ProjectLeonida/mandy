"""Read speaker slots and subtitle labels from typed PSIN conversation records."""

import collections
import re
import struct

from psin_data import Container


CONVERSATION = 0xEDAF33B4
PARTICIPANT = 0x1638FDAA
TURN = 0x06A04D25
LINE = 0x7622C7F9

# LSCH structure sizes and the (name hash, byte offset, member type) of each field read here.
_LAYOUT = {
    CONVERSATION: (88, {(0x827F53FD, 16, 0x52), (0xCDE324D0, 32, 0x52)}),
    PARTICIPANT: (20, {(0xB2A5A79B, 0, 0x4A), (0xBD6915B4, 4, 0x14)}),
    TURN: (64, {(0x53C6B42D, 0, 0x52), (0x59519816, 24, 0x15)}),
    LINE: (40, {(0x9624A3B6, 0, 0x42), (0x66FB3A21, 8, 0x42)}),
}
_NAME = re.compile(rb"[A-Za-z0-9_]{1,104}")
_CONVERSATION_HASH = struct.pack("<I", CONVERSATION)


def _matches_layout(container):
    for kind, (size, fields) in _LAYOUT.items():
        entry = container.structs.get(kind)
        if entry is None or entry["size"] != size:
            return False
        if not fields <= {(m["hash"], m["offset"], m["type"]) for m in entry["members"]}:
            return False
    return True


def _array(container, at, kind):
    """Offsets of the type-0x52 array's records in the pointed-to block."""
    count, capacity, reserved = struct.unpack_from("<HHI", container.blob, at + 8)
    if not count:
        return ()
    number, offset = container.pointer(at)
    stride, _ = _LAYOUT[kind]
    if count > capacity or reserved or not 1 <= number <= len(container.blocks):
        raise ValueError("invalid conversation array header")
    block_type, base, size = container.blocks[number - 1]
    if block_type != kind or offset % stride or offset + capacity * stride > size:
        raise ValueError("invalid conversation array target")
    return range(base + offset, base + offset + count * stride, stride)


def _name(container, at):
    number, offset = container.pointer(at)
    if not 1 <= number <= len(container.blocks):
        return None
    block_type, base, size = container.blocks[number - 1]
    if block_type != 0x10 or offset >= size:
        return None
    start = base + offset
    end = container.blob.find(b"\0", start, min(base + size, start + 105))
    if end == -1 or not _NAME.fullmatch(container.blob[start:end]):
        return None
    return container.blob[start:end].decode("ascii")


def speaker_rows(blob):
    """Yield native line locations, slots, hashes and labels from a supported container.

    Offsets are relative to the PSIN blob. A missing, repeated or zero-hash participant slot has no known speaker.
    Malformed arrays never borrow a record from an adjacent block or another conversation.
    """
    if blob[:4] != b"PSIN" or _CONVERSATION_HASH not in blob:
        return
    try:
        container = Container(blob)
    except (ValueError, struct.error):
        return
    if not container.fills or not _matches_layout(container):
        return
    for kind, base, size in container.blocks:
        stride, _ = _LAYOUT[CONVERSATION]
        if kind != CONVERSATION or size % stride:
            continue
        for conversation in range(base, base + size, stride):
            try:
                participants = _array(container, conversation + 16, PARTICIPANT)
                turns = _array(container, conversation + 32, TURN)
            except ValueError:
                continue
            by_slot = collections.defaultdict(list)
            for participant in participants:
                character_hash, slot = struct.unpack_from("<II", blob, participant)
                by_slot[slot].append(character_hash)
            for turn in turns:
                slot = struct.unpack_from("<I", blob, turn + 24)[0]
                characters = by_slot.get(slot, ())
                character_hash = characters[0] if len(characters) == 1 else 0
                try:
                    lines = _array(container, turn, LINE)
                except ValueError:
                    continue
                for line in lines:
                    cue = _name(container, line)
                    label = _name(container, line + 8)
                    if cue and label:
                        yield conversation, turn, line, slot, character_hash, cue, label
