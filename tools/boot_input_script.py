"""Parse LEONIDA_INPUT_SCRIPT for scripts/windows-boot/boot.ps1.

Absolute entries (``button@seconds``) go to the runtime as before. State-relative entries
(``button@game=3+5`` / ``button@session=d+5``) are returned to the harness, which watches the
``LEONIDA_PEEK`` values and injects the key once the state has been seen for the delay.

No game files, no Windows, no GPU: this module only parses and matches the script text.
"""

import argparse
import json
import math
import sys

BUTTONS = {
    "a", "b", "x", "y", "start", "back", "lb", "rb", "up", "down", "left", "right", "rsup", "rsdown",
}
# The keyboard pad maps keys to these buttons (docs/gameinput.md, "Keyboard pad"); rsup/rsdown are
# stick throws with no key, so the harness cannot inject them.
KEY_BUTTONS = {"a", "b", "x", "y", "start", "back", "lb", "rb", "up", "down", "left", "right"}
STATE_FIELDS = {"game", "session"}


def _number(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _hex(text):
    text = text.strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    try:
        int(text, 16)
    except (TypeError, ValueError):
        return None
    return text or None


def parse(text):
    """Return {"absolute": str, "state": [...], "rejected": [...]} for one script."""
    absolute = []
    state = []
    rejected = []
    for raw in (text or "").split(","):
        entry = raw.strip()
        if not entry:
            continue
        if "@" not in entry:
            rejected.append(entry)
            continue
        button, spec = entry.split("@", 1)
        button = button.strip().lower()
        spec = spec.strip()
        if button not in BUTTONS:
            rejected.append(entry)
            continue
        if spec.lower().startswith(("game=", "session=")):
            field, rest = spec.split("=", 1)
            field = field.strip().lower()
            if field not in STATE_FIELDS or button not in KEY_BUTTONS:
                rejected.append(entry)
                continue
            if "+" in rest:
                value_text, delay_text = rest.split("+", 1)
            else:
                value_text, delay_text = rest, "0"
            value = _hex(value_text)
            delay = _number(delay_text)
            if value is None or delay is None:
                rejected.append(entry)
                continue
            state.append({"button": button, "field": field, "value": value, "delay": delay})
        else:
            seconds = _number(spec)
            if seconds is None:
                rejected.append(entry)
                continue
            absolute.append("%s@%s" % (button, spec))
    return {"absolute": ",".join(absolute), "state": state, "rejected": rejected}


def due(plan, now, game=None, session=None, seen=None, fired=None):
    """Entries in ``plan`` whose state has been seen for its delay and that have not fired."""
    seen = seen or {}
    fired = set(fired or ())
    out = []
    for index, entry in enumerate(plan.get("state", [])):
        if index in fired:
            continue
        state = game if entry["field"] == "game" else session
        if state is None:
            continue
        state_value = _hex(str(state))
        if state_value is None or int(state_value, 16) != int(entry["value"], 16):
            continue
        since = seen.get(entry["field"])
        if since is None or now < since + entry["delay"]:
            continue
        out.append({"index": index, "button": entry["button"]})
    return out


def _load_plan(args):
    if args.plan:
        with open(args.plan, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if args.plan_json:
        return json.loads(args.plan_json)
    raise SystemExit("due needs --plan or --plan-json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    parse_cmd = sub.add_parser("parse", help="split a script into absolute and state-relative entries")
    parse_cmd.add_argument("script")
    parse_cmd.add_argument("--out", help="write the plan here as JSON")

    due_cmd = sub.add_parser("due", help="the entries whose state has been seen for their delay")
    due_cmd.add_argument("--plan")
    due_cmd.add_argument("--plan-json")
    due_cmd.add_argument("--now", type=float, required=True)
    due_cmd.add_argument("--game")
    due_cmd.add_argument("--session")
    due_cmd.add_argument("--seen-game", type=float)
    due_cmd.add_argument("--seen-session", type=float)
    due_cmd.add_argument("--fired", default="")

    args = parser.parse_args(argv)
    if args.command == "parse":
        plan = parse(args.script)
        text = json.dumps(plan, indent=2)
        if args.out:
            with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text + "\n")
        print(text)
        return 0

    plan = _load_plan(args)
    seen = {}
    if args.seen_game is not None:
        seen["game"] = args.seen_game
    if args.seen_session is not None:
        seen["session"] = args.seen_session
    fired = [int(part) for part in args.fired.split(",") if part.strip().isdigit()]
    print(json.dumps(due(plan, args.now, args.game, args.session, seen, fired)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
