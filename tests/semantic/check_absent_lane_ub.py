"""Static SPIR-V invariant checker for absent-lane / inactive-lane subgroup UB.

Verifies that every `OpGroupNonUniformShuffle` and `OpGroupNonUniformBroadcast` in a
SPIR-V module (assembly text or binary `.spv`) only ever passes a lane index provably
guaranteed to be a running invocation in the current subgroup:
  * own physical lane (`BuiltIn SubgroupLocalInvocationId`),
  * a lane proven present by `OpGroupNonUniformBallotBitExtract` on a subgroup ballot,
  * the lowest set bit of a non-empty subgroup ballot (or its 32-lane half word),
  * an `OpSelect` combining provably running lanes (never falling back to `%uint_0` or
    any constant / unguarded lane).

Also flags `OpGroupNonUniformQuadSwap` / `OpGroupNonUniformQuadBroadcast` in non-Fragment
stages (where partial quads are possible in Vulkan).

Standalone CLI usage:
    python3 -I tests/semantic/check_absent_lane_ub.py path/to/shader.spv [shader2.spvasm ...]

Standard library only; no GPU, no LLVM, no external SPIRV-Tools needed.
"""
import os
import re
import struct
import sys

_TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import spirv_asm_tables as _spv_tables  # noqa: E402

_LINE_RE = re.compile(r"^(?:(%\S+)\s*=\s*)?(Op\w+)\s*(.*)$")

# Derive all binary SPIR-V opcodes and enum values directly from tools/spirv_asm_tables.py.
_SPV_MAGIC = 0x07230203
_TRACKED_OPS = frozenset({
    "OpEntryPoint",
    "OpTypeInt",
    "OpTypeFloat",
    "OpTypeVector",
    "OpConstantTrue",
    "OpConstantFalse",
    "OpConstant",
    "OpConstantNull",
    "OpVariable",
    "OpLoad",
    "OpStore",
    "OpDecorate",
    "OpCompositeExtract",
    "OpCopyObject",
    "OpExtInst",
    "OpIAdd",
    "OpLogicalOr",
    "OpLogicalAnd",
    "OpSelect",
    "OpIEqual",
    "OpINotEqual",
    "OpBitwiseOr",
    "OpBitwiseXor",
    "OpBitwiseAnd",
    "OpGroupNonUniformAll",
    "OpGroupNonUniformAny",
    "OpGroupNonUniformBroadcast",
    "OpGroupNonUniformBroadcastFirst",
    "OpGroupNonUniformBallot",
    "OpGroupNonUniformInverseBallot",
    "OpGroupNonUniformBallotBitExtract",
    "OpGroupNonUniformBallotBitCount",
    "OpGroupNonUniformBallotFindLSB",
    "OpGroupNonUniformBallotFindMSB",
    "OpGroupNonUniformShuffle",
    "OpGroupNonUniformShuffleXor",
    "OpGroupNonUniformShuffleUp",
    "OpGroupNonUniformShuffleDown",
    "OpGroupNonUniformQuadBroadcast",
    "OpGroupNonUniformQuadSwap",
})
_OP_NAMES = {_spv_tables.INSTRUCTIONS[name][0]: name for name in _TRACKED_OPS}
_EXEC_MODELS = {}
for _em_name, (_em_code, _) in _spv_tables.VALUE_ENUMS["ExecutionModel"].items():
    _EXEC_MODELS.setdefault(_em_code, _em_name)
_DECORATION_BUILTIN = _spv_tables.VALUE_ENUMS["Decoration"]["BuiltIn"][0]
_BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID = _spv_tables.VALUE_ENUMS["BuiltIn"]["SubgroupLocalInvocationId"][0]
_GLSL_FIND_ILSB = _spv_tables.GLSL_STD_450["FindILsb"]


def _parse_spv_binary(data: bytes):
    """Parse the subset of a SPIR-V binary needed for static subgroup lane safety checks."""
    if len(data) < 20 or len(data) % 4 != 0:
        raise ValueError("invalid SPIR-V binary length")
    magic = struct.unpack_from("<I", data, 0)[0]
    endian = "<" if magic == _SPV_MAGIC else ">" if magic == 0x03022307 else None
    if endian is None:
        raise ValueError(f"bad SPIR-V magic: {magic:#x}")
    words = struct.unpack(f"{endian}{len(data) // 4}I", data)
    insns = []
    consts = {}
    builtins = {}
    exec_models = set()
    id_str = lambda i: f"%{i}"  # noqa: E731
    idx = 5
    while idx < len(words):
        wc = words[idx] >> 16
        op = words[idx] & 0xFFFF
        if wc == 0 or idx + wc > len(words):
            break
        args = words[idx + 1:idx + wc]
        idx += wc
        name = _OP_NAMES.get(op)
        if name is None:
            continue
        if name == "OpEntryPoint" and args:
            exec_models.add(_EXEC_MODELS.get(args[0], str(args[0])))
        elif name == "OpDecorate" and len(args) >= 3:
            if args[1] == _DECORATION_BUILTIN and args[2] == _BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID:
                builtins[id_str(args[0])] = "SubgroupLocalInvocationId"
        elif name == "OpConstant" and len(args) >= 3:
            consts[id_str(args[1])] = args[2]
        elif name == "OpConstantNull" and len(args) >= 2:
            consts[id_str(args[1])] = 0
        elif name == "OpConstantTrue" and len(args) >= 2:
            consts[id_str(args[1])] = True
        elif name == "OpConstantFalse" and len(args) >= 2:
            consts[id_str(args[1])] = False
        elif name == "OpStore" and len(args) >= 2:
            insns.append((None, name, [id_str(args[0]), id_str(args[1])]))
        elif name == "OpExtInst" and len(args) >= 5:
            ext_op = "FindILsb" if args[3] == _GLSL_FIND_ILSB else str(args[3])
            insns.append((id_str(args[1]), name, [id_str(args[0]), id_str(args[2]), ext_op] + [id_str(x) for x in args[4:]]))
        elif name == "OpCompositeExtract" and len(args) >= 4:
            insns.append((id_str(args[1]), name, [id_str(args[0]), id_str(args[2]), str(args[3])]))
        elif len(args) >= 2:
            # Result-bearing instructions have [result_type, result_id, ...operands]
            insns.append((id_str(args[1]), name, [id_str(args[0])] + [id_str(x) for x in args[2:]]))
    return insns, consts, builtins, exec_models


def _parse_spv_text(text: str):
    insns = []
    consts = {}
    builtins = {}
    exec_models = set()
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        res, op, rest = m.groups()
        a = rest.split()
        if op == "OpEntryPoint" and a:
            exec_models.add(a[0])
        elif op == "OpDecorate" and len(a) >= 3 and a[1] == "BuiltIn":
            builtins[a[0]] = a[2]
        elif op == "OpConstant" and res and len(a) >= 2:
            try:
                consts[res] = int(a[1], 0)
            except ValueError:
                pass
        elif op == "OpConstantNull" and res:
            consts[res] = 0
        elif op == "OpConstantTrue" and res:
            consts[res] = True
        elif op == "OpConstantFalse" and res:
            consts[res] = False
        else:
            insns.append((res, op, a))
    return insns, consts, builtins, exec_models


def check_absent_lane_ub(module) -> list:
    """Return a list of violation strings for any unsafe subgroup shuffle/broadcast/quad op.

    `module` may be SPIR-V assembly text (`str`) or SPIR-V binary (`bytes` / `bytearray`).
    An empty list means every shuffle/broadcast lane operand is statically proven to name a
    running invocation (never `%uint_0` or an unguarded lane), and no quad op appears in a
    non-Fragment stage.
    """
    if isinstance(module, (bytes, bytearray)):
        insns, consts, builtins, exec_models = _parse_spv_binary(bytes(module))
    else:
        insns, consts, builtins, exec_models = _parse_spv_text(module)

    defs = {}
    var_single_store = {}
    var_store_count = {}
    for res, op, a in insns:
        if res:
            defs[res] = (op, a)
        elif op == "OpStore" and len(a) >= 2:
            ptr, val = a[0], a[1]
            var_store_count[ptr] = var_store_count.get(ptr, 0) + 1
            var_single_store[ptr] = val

    sg_lane_vars = {vid for vid, b in builtins.items() if b == "SubgroupLocalInvocationId"}

    def unwrap(x, depth=0):
        """Follow OpCopyObject and single-store Function variable (Gen.eager) loads."""
        if depth > 16 or x not in defs:
            return x
        op, a = defs[x]
        if op == "OpCopyObject" and len(a) >= 2:
            return unwrap(a[1], depth + 1)
        if op == "OpLoad" and len(a) >= 2 and var_store_count.get(a[1]) == 1:
            return unwrap(var_single_store[a[1]], depth + 1)
        return x

    def is_const(x, val):
        x = unwrap(x)
        return x in consts and consts[x] == val

    def is_sg_lane(x):
        x = unwrap(x)
        if x in defs:
            op, a = defs[x]
            if op == "OpLoad" and len(a) >= 2 and a[1] in sg_lane_vars:
                return True
        return False

    def is_half_base(x):
        """Matches `sg_lane & 32`."""
        x = unwrap(x)
        if x not in defs:
            return False
        op, a = defs[x]
        if op == "OpBitwiseAnd" and len(a) >= 3:
            return (is_sg_lane(a[1]) and is_const(a[2], 32)) or (is_sg_lane(a[2]) and is_const(a[1], 32))
        return False

    def is_half_local_self(x):
        """Matches `sg_lane & 31`."""
        x = unwrap(x)
        if x not in defs:
            return False
        op, a = defs[x]
        if op == "OpBitwiseAnd" and len(a) >= 3:
            return (is_sg_lane(a[1]) and is_const(a[2], 31)) or (is_sg_lane(a[2]) and is_const(a[1], 31))
        return False

    def unpack_wave_source_lane(x):
        """If `x` is `(sg_lane & 32) | (inner & 31)` (or `(sg_lane & 32) | inner`), return `inner`, else None."""
        x = unwrap(x)
        if x not in defs:
            return None
        op, a = defs[x]
        if op == "OpBitwiseOr" and len(a) >= 3:
            if is_half_base(a[1]):
                rhs = unwrap(a[2])
            elif is_half_base(a[2]):
                rhs = unwrap(a[1])
            else:
                return None
            if rhs in defs and defs[rhs][0] == "OpBitwiseAnd" and len(defs[rhs][1]) >= 3:
                ra = defs[rhs][1]
                if is_const(ra[2], 31):
                    return ra[1]
                if is_const(ra[1], 31):
                    return ra[2]
            return rhs
        return None

    def ballot_of_word(w):
        """If `w` is an extracted 32-bit word of a ballot (directly or via wave32_ballot_word), return (ballot_id, half)."""
        w = unwrap(w)
        if w not in defs:
            return None
        op, a = defs[w]
        if op == "OpCompositeExtract" and len(a) >= 3:
            src = unwrap(a[1])
            if src in defs and defs[src][0] == "OpGroupNonUniformBallot":
                return (src, int(a[2]))
        if op == "OpSelect" and len(a) >= 4:
            # wave32_ballot_word: OpSelect %uint upper hi lo where hi=bal[1], lo=bal[0]
            b1 = ballot_of_word(a[2])
            b0 = ballot_of_word(a[3])
            if b1 and b0 and b1[0] == b0[0] and {b1[1], b0[1]} == {0, 1}:
                return (b1[0], "half")
        return None

    def ballot_words_nonzero(cond):
        """Return the ballot id if `cond` tests that a ballot (or its half word) is non-zero, else None."""
        cond = unwrap(cond)
        if cond not in defs:
            return None
        op, a = defs[cond]
        if op == "OpGroupNonUniformAny" and len(a) >= 3:
            return ("any_pred", unwrap(a[2]))
        if op == "OpINotEqual" and len(a) >= 3:
            bw = ballot_of_word(a[1]) if is_const(a[2], 0) else ballot_of_word(a[2]) if is_const(a[1], 0) else None
            if bw is not None:
                return ("ballot", bw[0], bw[1])
        if op == "OpLogicalOr" and len(a) >= 3:
            b1 = ballot_words_nonzero(a[1])
            b2 = ballot_words_nonzero(a[2])
            if b1 and b2 and b1[0] == "ballot" and b2[0] == "ballot" and b1[1] == b2[1]:
                return ("ballot", b1[1], "multi")
        return None

    def ballot_of_lsb(x):
        """If `x` is the LSB of a ballot (BallotFindLSB, FindILsb of half word, or ballot_lsb_by_words), return info."""
        x = unwrap(x)
        if x not in defs:
            return None
        op, a = defs[x]
        if op == "OpGroupNonUniformBallotFindLSB" and len(a) >= 3:
            bal = unwrap(a[2])
            if bal in defs and defs[bal][0] == "OpGroupNonUniformBallot":
                return ("phys", bal, unwrap(defs[bal][1][2]))
        if op == "OpExtInst" and len(a) >= 4 and a[2] == "FindILsb":
            bw = ballot_of_word(a[3])
            if bw is not None:
                bal = bw[0]
                return ("half" if bw[1] == "half" else "word", bal, unwrap(defs[bal][1][2]))
        if op == "OpIAdd" and len(a) >= 3:
            inner = ballot_of_lsb(a[1]) or ballot_of_lsb(a[2])
            if inner is not None and inner[0] == "word":
                return inner
        if op == "OpSelect" and len(a) >= 4:
            # ballot_lsb_by_words folds words 3..0 with OpSelect(w != 0, lsb_k, prev)
            nz = ballot_words_nonzero(a[1])
            lt = ballot_of_lsb(a[2])
            lf = ballot_of_lsb(a[3])
            if nz and nz[0] == "ballot" and lt and lf and lt[1] == nz[1] and lf[1] == nz[1]:
                return ("phys", nz[1], lt[2])
        return None

    def ballot_extract_lane(cond):
        """If `cond` implies lane `L` is set in a subgroup ballot, return the set of proven-running lane ids."""
        cond = unwrap(cond)
        if cond not in defs:
            return set()
        op, a = defs[cond]
        if op == "OpGroupNonUniformBallotBitExtract" and len(a) >= 4:
            bal = unwrap(a[2])
            if bal in defs and defs[bal][0] == "OpGroupNonUniformBallot":
                return {unwrap(a[3])}
        if op == "OpLogicalAnd" and len(a) >= 3:
            return ballot_extract_lane(a[1]) | ballot_extract_lane(a[2])
        return set()

    def is_safe_half_local(x, depth=0):
        """True if `x` is a logical 0..31 lane guaranteed to map via wave_source_lane to a running lane."""
        if depth > 24:
            return False
        x = unwrap(x)
        if is_half_local_self(x):
            return True
        if x not in defs:
            return False
        op, a = defs[x]
        if op == "OpSelect" and len(a) >= 4:
            cond, t_arm, f_arm = a[1], a[2], a[3]
            if not is_safe_half_local(f_arm, depth + 1):
                return False
            if is_safe_half_local(t_arm, depth + 1):
                return True
            nz = ballot_words_nonzero(cond)
            lsb = ballot_of_lsb(t_arm)
            if nz and lsb and nz[0] == "ballot" and nz[1] == lsb[1] and lsb[0] == "half":
                return True
        return False

    def is_safe_physical(x, depth=0):
        """True if `x` is statically proven to be a running physical SubgroupLocalInvocationId."""
        if depth > 24:
            return False
        x = unwrap(x)
        if is_sg_lane(x):
            return True
        inner = unpack_wave_source_lane(x)
        if inner is not None and is_safe_half_local(inner, depth + 1):
            return True
        if x not in defs:
            return False
        op, a = defs[x]
        if op == "OpSelect" and len(a) >= 4:
            cond, t_arm, f_arm = a[1], a[2], a[3]
            if not is_safe_physical(f_arm, depth + 1):
                return False
            if is_safe_physical(t_arm, depth + 1):
                return True
            # Guarded by BallotBitExtract(bal, t_arm)
            proven = ballot_extract_lane(cond)
            if unwrap(t_arm) in proven:
                return True
            # Guarded by ballot non-zero -> LSB of that same ballot
            nz = ballot_words_nonzero(cond)
            lsb = ballot_of_lsb(t_arm)
            if nz and lsb and lsb[0] == "phys":
                if nz[0] == "ballot" and nz[1] == lsb[1]:
                    return True
                if nz[0] == "any_pred" and nz[1] == lsb[2]:
                    return True
        return False

    def describe_reason(x):
        ux = unwrap(x)
        if ux in consts:
            return f"constant lane {consts[ux]} ({ux})"
        if ux in defs and defs[ux][0] == "OpSelect":
            a = defs[ux][1]
            f_arm = unwrap(a[3])
            if f_arm in consts:
                return f"OpSelect {ux} falls back to constant lane {consts[f_arm]} ({f_arm})"
            return f"OpSelect {ux} has unverified arm(s) (true={a[2]}, false={a[3]})"
        return f"unguarded lane operand {ux}"

    violations = []
    for res, op, a in insns:
        if op in ("OpGroupNonUniformShuffle", "OpGroupNonUniformBroadcast") and len(a) >= 4:
            lane_id = a[3]
            if not is_safe_physical(lane_id):
                violations.append(f"{res} = {op} ... {lane_id}: {describe_reason(lane_id)}")
        elif op in ("OpGroupNonUniformShuffleXor", "OpGroupNonUniformShuffleUp", "OpGroupNonUniformShuffleDown"):
            violations.append(f"{res} = {op}: relative/xor shuffle can read absent lanes in a partial subgroup")
        elif op in ("OpGroupNonUniformQuadSwap", "OpGroupNonUniformQuadBroadcast"):
            if exec_models and not exec_models.issubset({"Fragment"}):
                violations.append(f"{res} = {op} in non-Fragment stage {sorted(exec_models)} (partial-quad UB)")
    return violations


def main(argv=None):
    paths = sys.argv[1:] if argv is None else argv
    if not paths:
        print("usage: check_absent_lane_ub.py <file.spv|file.spvasm> [...]", file=sys.stderr)
        return 2
    rc = 0
    for path in paths:
        with open(path, "rb") as f:
            raw = f.read()
        mod = raw if len(raw) >= 4 and struct.unpack_from("<I", raw, 0)[0] in (_SPV_MAGIC, 0x03022307) else raw.decode("utf-8")
        viols = check_absent_lane_ub(mod)
        if viols:
            rc = 1
            print(f"FAIL {path}: {len(viols)} absent-lane UB site(s):")
            for v in viols[:20]:
                print(f"  - {v}")
        else:
            print(f"OK   {path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
