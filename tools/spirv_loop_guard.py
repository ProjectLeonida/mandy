"""Bound each structured SPIR-V loop, including the translator's PC dispatch and wave schedulers.

Five-star-recomp b12a952 identified nonterminating translated loops during story loading.
Instrument the back edge so a normally terminating iteration takes its original exit.

Each loop counts its own visits from every entry, so nested loops multiply their limits: an inner loop may make
65,536 visits in each of 65,536 visits of the outer one, which bounds nothing a GPU survives. `total` adds a second
count, of every guarded back edge of an invocation, in a Private variable, which no back edge passes once it reaches
the total. Each loop keeps its own limit; a loop that ends by itself is cut only after the whole invocation has made
`total` visits. The case it is for: a loop bound computed by a float divide whose divisor is 0 is +inf, converts to
0xFFFFFFFF, and an inclusive exit test against it never fails once the counter wraps; nested loops of that kind ran
for seconds a dispatch at the per-loop limit, and the device was lost.
"""
import argparse
from collections import defaultdict
import re

DEFAULT_LIMIT = 65536
DEFAULT_TOTAL = 1 << 20     # visits of all the guarded loops of an invocation together
_LINE = re.compile(r"(?m)^[ \t]*(?:(%[\w.]+)\s*=\s*)?(Op(?:Label|BranchConditional|Branch|Switch|"
                   r"LoopMerge|SelectionMerge|Phi|Function|TypeInt|TypeBool|TypePointer|Constant|"
                   r"EntryPoint))\b([^;\n]*)")


def parse_limit(text):
    value = int(text)
    if not 0 <= value <= 0xffffffff:
        raise argparse.ArgumentTypeError("loop limit must be in 0..4294967295")
    return value


def add_guards(text, limit=DEFAULT_LIMIT, total=DEFAULT_TOTAL):
    if not 0 <= limit <= 0xffffffff:
        raise ValueError("loop limit must fit an unsigned 32-bit word")
    if not 0 <= total <= 0xffffffff:
        raise ValueError("loop total must fit an unsigned 32-bit word")
    if not limit or "OpLoopMerge" not in text:
        return text
    prefix = "%loop_guard_"
    while prefix in text:
        prefix += "_"
    heads, labels, terminators, merges, phis = {}, {}, {}, {}, defaultdict(list)
    pred = defaultdict(list)
    constants, pointers = {}, {}
    uint = boolean = first_function = block = entry_point = None
    for match in _LINE.finditer(text):
        result, op, operands = match.groups()
        args = operands.split()
        if op == "OpFunction":
            if first_function is None:
                first_function = match.start()
        elif op == "OpEntryPoint":
            entry_point = match.end()
        elif op == "OpTypePointer":
            pointers[tuple(args)] = result
        elif op == "OpTypeInt" and args == ["32", "0"]:
            uint = result
        elif op == "OpTypeBool":
            boolean = result
        elif op == "OpConstant" and args[0] == uint:
            constants[int(args[1], 0)] = result
        elif op == "OpLabel":
            block = result
            labels[block] = match.end()
        elif op == "OpPhi":
            phis[block].append((match.end(), args[0]))
        elif op in ("OpLoopMerge", "OpSelectionMerge"):
            merges[block] = match.start()
            if op == "OpLoopMerge":
                heads[block] = args[0]
        elif op in ("OpBranch", "OpBranchConditional", "OpSwitch"):
            terminators[block] = (match.start(), match.end(), op, args)
            targets = (args[:1] if op == "OpBranch" else args[1:3] if op == "OpBranchConditional"
                       else args[1::2])
            for dest in dict.fromkeys(targets):
                pred[dest].append(block)
    if not heads:
        return text
    decls = []
    if uint is None:
        uint = prefix + "uint"
        decls.append(f"{uint} = OpTypeInt 32 0")
    if boolean is None:
        boolean = prefix + "bool"
        decls.append(f"{boolean} = OpTypeBool")
    # The invocation's total is a global, which the entry point lists (SPIR-V 1.4 lists every global it uses); a
    # fragment without an entry point keeps the per-loop counts alone.
    totalled = total > 0 and entry_point is not None
    for value in (1, limit) + ((0, total) if totalled else ()):
        if value not in constants:
            constants[value] = prefix + f"c{value}"
            decls.append(f"{constants[value]} = OpConstant {uint} {value}")

    edits, undef = [], {}
    if totalled:
        private = pointers.get(("Private", uint))
        if private is None:
            private = prefix + "private"
            decls.append(f"{private} = OpTypePointer Private {uint}")
        decls.append(f"{prefix}total = OpVariable {private} Private {constants[0]}")
        edits.append((entry_point, entry_point, f" {prefix}total"))
    for i, (header, exit_label) in enumerate(heads.items()):
        # The translator emits dominators first. A structured loop has exactly one back edge.
        backward = [p for p in pred[header] if labels[p] >= labels[header]]
        if len(backward) != 1:
            raise ValueError(f"expected one back edge at {header}, found {len(backward)}")
        back = backward[0]
        q = f"{prefix}{i}_"
        start, end, op, args = terminators[back]
        if op == "OpSwitch":
            raise ValueError(f"unexpected switch on the back edge of {header}")
        # Split a conditional back edge when its other target is inside the construct.
        # This keeps that edge, and therefore its original condition, independent of the budget.
        split = op == "OpBranchConditional" and exit_label not in args[1:3]
        latch = q + "latch" if split else back
        incoming = " ".join((q + "next " + latch) if p == back else (constants[1] + " " + p) for p in pred[header])
        edits.append((labels[header], labels[header], f"\n{q}count = OpPhi {uint} {incoming}"))
        operations = [f"{q}next = OpIAdd {uint} {q}count {constants[1]}"]
        if totalled:
            # Room in this loop's own count and in the invocation's total: the back edge needs both.
            operations += [f"{q}spent = OpLoad {uint} {prefix}total",
                           f"{q}spent_next = OpIAdd {uint} {q}spent {constants[1]}",
                           f"OpStore {prefix}total {q}spent_next",
                           f"{q}own = OpULessThan {boolean} {q}count {constants[limit]}",
                           f"{q}left = OpULessThan {boolean} {q}spent {constants[total]}",
                           f"{q}room = OpLogicalAnd {boolean} {q}own {q}left"]
        else:
            operations.append(f"{q}room = OpULessThan {boolean} {q}count {constants[limit]}")
        if op == "OpBranch" or split:
            branch = f"OpBranchConditional {q}room {header} {exit_label}"
        else:
            if args[1] == header:
                operations.append(f"{q}take = OpLogicalAnd {boolean} {args[0]} {q}room")
            else:
                operations.extend([f"{q}full = OpLogicalNot {boolean} {q}room",
                                   f"{q}take = OpLogicalOr {boolean} {args[0]} {q}full"])
            branch = f"OpBranchConditional {q}take {args[1]} {args[2]}"
        if split:
            args = [latch if a == header else a for a in args]
            operations[:0] = ["OpBranchConditional " + " ".join(args), latch + " = OpLabel"]
        pos = start if split else merges.get(back, start)
        edits.append((pos, pos, "\n".join(operations) + "\n"))
        edits.append((start, end, branch))
        if exit_label not in terminators[back][3][1:3]:
            for at, ty in phis[exit_label]:
                if ty not in undef:
                    undef[ty] = prefix + "undef" + str(len(undef))
                    decls.append(f"{undef[ty]} = OpUndef {ty}")
                edits.append((at, at, f" {undef[ty]} {latch}"))
        if split:
            # All original header phis now receive their back-edge values from the split latch.
            for at, _ in phis[header]:
                line_start = text.rfind("\n", 0, at) + 1
                for match in re.finditer(r"(?<!\S)" + re.escape(back) + r"(?=\s|$)", text[line_start:at]):
                    edits.append((line_start + match.start(), line_start + match.end(), latch))
    if decls:
        edits.append((first_function, first_function, "\n".join(decls) + "\n"))
    chunks, previous = [], 0
    for start, end, replacement in sorted(edits, key=lambda e: (e[0], e[1])):
        chunks.extend((text[previous:start], replacement))
        previous = end
    chunks.append(text[previous:])
    return "".join(chunks)
