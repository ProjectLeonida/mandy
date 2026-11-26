"""tests/semantic/piecewise_parse.py builds the tree a whole-file parse builds, and support.py's loader the same code.

    python -I tests/test_piecewise_parse.py

The translator tests load tools/rdna2spirv.py through that loader, so "the same" is the whole point: a test that ran a
slightly different translator would prove nothing. Checked here, with nothing but the standard library:

* hand-written sources for each boundary rule (decorators, the clauses of `if`/`try`/`for`/`while` at top level and
  in a class, a class header over several lines, nested classes, a docstring and `from __future__`, strings and
  brackets whose inner lines start at column 0, non-ASCII text, CRLF), cut into pieces of 1, 2, 3, 5 and 8 lines:
  `ast.dump(tree, include_attributes=True)` equals the whole-file parse's;
* sources with an error: the pieces are given up and `parse` raises what `compile` raises (type, message, line,
  column, text);
* every Python file under tools/, scripts/ and tests/ and make_game.py in pieces of one statement, and the long
  files of tools/ also in pieces of the size the loader uses: the two trees agree in every field and position
  (`same_tree`, which is held against the dumps on the hand-written sources);
* the code object compiled from the pieces against the one `compile` makes from the file's bytes, for some files;
* tools/rdna2spirv.py, tree (by its dump) and code object, in one test of its own: on the interpreters the pieces
  are for (CPython 3.12 to 3.14.7) each of its two whole-file parses takes 7 s, which is why it is `_slow`;
* support.py's finder and loader on throwaway folders, and that support.py installs them only where needed.

About 45 s in all on those interpreters, most of it the repository scan and the translator.
"""
import ast
import importlib.machinery
import importlib.util
import os
import py_compile
import sys
import tempfile
import types
import unittest
import warnings
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEMANTIC = os.path.join(ROOT, "tests", "semantic")
TOOLS = os.path.join(ROOT, "tools")
TRANSLATOR = os.path.join(TOOLS, "rdna2spirv.py")
sys.path.insert(0, SEMANTIC)

import piecewise_parse as pp  # noqa: E402
import support  # noqa: E402  (imports the translator the way the translator tests do)

SPLITS = sys.version_info >= (3, 12)        # older interpreters are always parsed whole (piecewise_parse.parse_pieces)
SIZES = (1, 2, 3, 5, 8)
DUMP = dict(include_attributes=True, **({"show_empty": True} if sys.version_info >= (3, 13) else {}))


def whole(source, filename="<test>"):
    return compile(source, filename, "exec", ast.PyCF_ONLY_AST, dont_inherit=True)


def dump(tree):
    return ast.dump(tree, **DUMP)


def difference(got, expected):
    """Where two dumps part, with a little of each: the dumps themselves run to megabytes."""
    at = len(os.path.commonprefix([got, expected]))
    near = slice(max(0, at - 90), at + 90)
    return f"dumps differ at character {at}: ...{got[near]!r} against ...{expected[near]!r}"


def same_tree(a, b):
    """Whether two trees agree in every field and every position: what equal dumps say, without building them.
    The dumps of the repository's 24 MB of Python take 12 s, this takes 3; PiecesTest holds it against `dump`."""
    todo = [(a, b)]
    while todo:
        x, y = todo.pop()
        if type(x) is not type(y):
            return False
        if isinstance(x, ast.AST):
            todo.extend((getattr(x, name, None), getattr(y, name, None)) for name in x._fields + x._attributes)
        elif type(x) is list:
            if len(x) != len(y):
                return False
            todo.extend(zip(x, y))
        elif (repr(x) != repr(y)) if isinstance(x, (float, complex)) else (x != y):
            return False
    return True


def error_facts(call, *args):
    """What the SyntaxError `call` raises says (a ValueError before 3.12 for a NUL), or None when it raises none."""
    try:
        call(*args)
    except SyntaxError as e:
        return type(e).__name__, e.msg, e.filename, e.lineno, e.offset, e.end_lineno, e.end_offset, e.text
    except ValueError as e:
        return type(e).__name__, str(e)
    return None


def outcome(parse, *args):
    """What `parse` makes of a source: the dump of its tree, or what the error it raises says."""
    try:
        return dump(parse(*args))
    except (SyntaxError, ValueError):
        return error_facts(parse, *args)


def code_facts(code):
    """Everything a code object holds as plain values, the code objects among its constants included."""
    return (code.co_name, code.co_qualname, code.co_filename, code.co_firstlineno, code.co_flags, code.co_argcount,
            code.co_posonlyargcount, code.co_kwonlyargcount, code.co_nlocals, code.co_stacksize, code.co_code,
            code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars, code.co_linetable,
            code.co_exceptiontable, list(code.co_positions()),
            [code_facts(c) if isinstance(c, types.CodeType) else (type(c).__name__, repr(c)) for c in code.co_consts])


def python_files():
    """Every .py file under tools/, scripts/ and tests/, and make_game.py."""
    yield os.path.join(ROOT, "make_game.py")
    for top in ("tools", "scripts", "tests"):
        for base, dirs, names in os.walk(os.path.join(ROOT, top)):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not d.startswith("."))
            for name in sorted(names):
                if name.endswith(".py"):
                    yield os.path.join(base, name)


def read(path):
    with open(path, "rb") as f:
        return f.read()


# --- hand-written sources, one per boundary rule ---------------------------------------------------------------

DECORATORS = r'''
import functools


def deco(f):
    return f


@deco
def a():
    return 1


@deco
@functools.wraps(a)
def b():
    return 2


@deco
class C:
    @staticmethod
    def m():
        return 3

    @property
    @deco
    def p(self):
        return 4

    @functools.lru_cache(
        maxsize=None,
    )
    def q(self):
        return 5

    @deco
    class Inner:
        pass

    tail = 6
x = 1
'''

CLAUSES_AT_TOP_LEVEL = r'''
import os
if os.sep == "/":
    a = 1
elif os.sep == ":":
    a = 2
else:
    a = 3
try:
    b = int("x")
except ValueError:
    b = 0
except (TypeError, KeyError) as e:
    b = 1
else:
    b = 2
finally:
    c = 3
try:
    d = 4
finally:
    d = 5
for i in range(3):
    pass
else:
    e = 6
while False:
    pass
else:
    f = 7
with open(os.devnull) as g:
    pass
if a: b = 1
elif b: b = 2
else: b = 3
try: h = 8
except Exception: h = 9
i = 0; j = 1; k = 2
'''

CLAUSES_IN_A_CLASS = r'''
class K:
    if True:
        a = 1
    elif False:
        a = 2
    else:
        a = 3
    try:
        b = 1
    except Exception:
        b = 2
    else:
        b = 3
    finally:
        c = 4
    for i in ():
        pass
    else:
        d = 5
    while False:
        pass
    else:
        e = 6

    def m(self):
        if self:
            return 1
        else:
            return 2
        try:
            pass
        finally:
            pass

    async def n(self):
        async with self as s:
            async for t in s:
                await t
    f = 7; g = 8
    h = 9;
last = K
'''

MATCH_AND_EXCEPT_STAR = r'''
def f(x):
    match x:
        case 1:
            return "one"
        case [a, b]:
            return a
        case _:
            return None


match = 1
case = match
match case:
    case 1:
        pass


class M:
    match case:
        case 1:
            y = 1
        case _:
            y = 2
    case = 3
    try:
        z = 1
    except* ValueError:
        z = 2
    except* TypeError:
        z = 3
after = M
'''

CLASS_HEADER_OVER_SEVERAL_LINES = r'''
class Base:
    pass


class A(
    Base,
    metaclass=type,
):
    """Doc."""
    x = 1

    def m(self):
        return self.x

    y = 2


class B(Base,
        object): z = 3


class C(
        Base):
    def n(self): return 4
    w = 5
'''

NESTED_CLASSES = r'''
class Outer:
    a = 1

    class Middle:
        b = 2

        class Inner:
            c = 3

            def m(self):
                class Local:
                    d = 4
                return Local

        e = 5

    f = 6

    class Second: pass

    g = 7


class After(Outer.Middle):
    pass
'''

DOCSTRING_AND_FUTURE = r'''"""Module docstring.

class NotCode:
    else:
"""
from __future__ import annotations
from __future__ import division, generators

import sys


def f(a: Undefined) -> AlsoUndefined:
    """Function docstring."""
    return a


class D:
    """Class docstring."""

    x: NotDefinedEither = 1
'''

COLUMN_ZERO = r'''
text = """
class NotAClass:
    def not_a_method(self):
else:
@not_a_decorator
"""
value = (
1,
2,
[
3,
],
)
call = dict(
a=1,
)
other = text + \
"tail"
if value: first = {
"k": 1,
}
else: first = None


class Holder:
    doc = """
class StillNotAClass:
    pass
finally:
"""
    tup = (
1,
2,
)

    def method(self):
        return f"""
{self.doc}
{
self.tup
!r:>{
10
}}
""" + \
"x"
# a comment at column 0 inside the class body
  # and one that fits no indentation level

    after = 3
    lst = [
# a comment inside brackets
4,
]


last = Holder
'''

F_STRINGS = r'''
name = "x"
plain = f"{name}"
spec = f"{name!r:>{10}} {name=} {{literal}} {3.14:.{2}f}"
joined = (f"{name}"
          f"{name:{name}}"
          "tail")


class F:
    def m(self, a, b):
        self.emit(f"{a} = OpIAdd %uint {b} {self.cu(1)}")
        return f"""{a}
  {b}""", rf"\d{a}", f'{a["k"]}'

    table = {f"k{i}": f"{i:02x}" for i in range(4)}
'''

LEFTOVERS = r'''#!/usr/bin/env python3
# a comment before anything

x = 1  # trailing comment
lam = lambda: (yield)
star, *rest = 1, 2, 3
print(x); print(x)
global_name: int
assert x, "message"
del star
if x:
    pass
# comment after the last statement
'''

NO_FINAL_NEWLINE = "x = 1\nclass A:\n    y = 2\n    z = 3"
ONLY_COMMENTS = "# nothing\n\n# here\n"
BACKSLASH_LINES = "a = 1\n\\\nb = 2\nclass A:\n    c = 3\n\\\n    d = 4\n    e = 5\nf = 6\n"
FORM_FEED = "a = 1\n\x0c\nb = 2\nclass A:\n    c = 3\n\x0c    d = 4\ne = 5\n"
NON_ASCII = ('caf\xe9 = "\u017c\xf3\u0142w"; after = caf\xe9\n'
             'class \xdc:\n'
             '    \u03c0 = 3; q = "\U0001f600" + f"{\u03c0}\u2713"; r = q\n'
             '    def m(self): return self.\u03c0, "\xe9"\n'
             '    s = "\xe9" + \\\n'
             '        "\u03c0"; t = s\n'
             'last = "\xe9"; end = last\n')
LATIN_1 = ('# -*- coding: latin-1 -*-\n'
           'a = "\xe9\xe8"; b = a\n'
           'class L:\n'
           '    c = "\xfc"; d = c\n'
           '    def m(self): return "\xdf", self.c\n')

SOURCES = {
    "decorators": DECORATORS,
    "clauses at top level": CLAUSES_AT_TOP_LEVEL,
    "clauses in a class": CLAUSES_IN_A_CLASS,
    "tabs": CLAUSES_IN_A_CLASS.replace("    ", "\t"),
    "match and except*": MATCH_AND_EXCEPT_STAR,
    "class header over several lines": CLASS_HEADER_OVER_SEVERAL_LINES,
    "nested classes": NESTED_CLASSES,
    "docstring and __future__": DOCSTRING_AND_FUTURE,
    "strings and brackets at column 0": COLUMN_ZERO,
    "f-strings": F_STRINGS,
    "leftovers": LEFTOVERS,
    "no final newline": NO_FINAL_NEWLINE,
    "form feed": FORM_FEED,
    "non-ASCII": NON_ASCII,
}
if sys.version_info >= (3, 12):
    SOURCES["type parameters"] = r'''
type Pair[T] = tuple[T, T]


def first[T](pair: Pair[T]) -> T:
    return pair[0]


class Box[T]:
    type Inner = list[T]

    def get[U](self, other: U) -> T | U:
        return other

    nested = f"{'a' + f'{1}'}"
last = Box
'''

# A source with an error, and why it is one. None of them may come back as a tree.
ERRORS = {
    "bracket never closed": "a = 1\nb = (\nc = 3\nd = 4\n",
    "else without if": "a = 1\nelse:\n    b = 2\n",
    "else first": "else:\n    pass\n",
    "elif after a class": "class A:\n    a = 1\n    b = 2\nelif a:\n    pass\n",
    "indented first line": "    a = 1\nb = 2\n",
    "unexpected indent": "a = 1\n    b = 2\nc = 3\n",
    "unindent matches no level": "if a:\n        b = 1\n    c = 2\nd = 3\n",
    "unindent matches no level in a class": "class A:\n    a = 1\n  b = 2\n",
    "tabs against spaces": "if a:\n        b = 1\n\tc = 2\n",
    "tabs against spaces between class statements": "class A:\n        a = 1\n\tb = 2\n",
    "string never closed": 'a = 1\nb = """open\nc = 3\n',
    "error in a class body": "class A:\n    a = 1\n    b = = 2\n    c = 3\nd = 4\n",
    "error in the last statement of a class": "class A:\n    a = 1\n    b = 2\n    def m(self) return 3\n",
    "except without try in a class": "class A:\n    a = 1\n    except E:\n        pass\n",
    "decorator with nothing after it": "a = 1\n@dec\n",
    "decorator before an assignment": "@dec\na = 1\nb = 2\n",
    "decorator before an assignment in a class": "class A:\n    a = 1\n    @dec\n    b = 2\n    c = 3\n",
    "character that is no token": "a = 1\nb = $\nc = 3\n",
    "backslash at the end": "a = 1\nb = 2 \\",
    "backslash line at the end": "a = 1\n\\\n",
    "NUL": "a = 1\0\nb = 2\n",
    "try without except": "try:\n    a = 1\nb = 2\n",
    "class without a name": "a = 1\nclass:\n    pass\n",
    "missing colon": "a = 1\ndef f()\n    return 1\n",
    "f-string conversion": 'a = 1\nb = f"{a!z}"\nc = 3\n',
    "f-string never closed": 'a = 1\nb = f"{a\nc = 3\n',
    "assignment to a literal": "a = 1\n1 = a\nb = 2\n",
    "statement after a one-line class": "class A: a = 1\n    b = 2\n",
}

# What compile refuses after parsing (an unknown feature already while parsing, from 3.14). Parsing in pieces and
# compiling the tree says the same.
COMPILE_ERRORS = {
    "return outside a function": "def f():\n    pass\nreturn 1\n",
    "return in a class": "class A:\n    a = 1\n    return 2\n    b = 3\n",
    "__future__ after code": "a = 1\nfrom __future__ import annotations\nb = 2\n",
    "break outside a loop": "a = 1\nbreak\n",
    "nonlocal at module level": "a = 1\nnonlocal a\n",
    "unknown __future__ feature": "from __future__ import no_such_feature\na = 1\n",
    "from __future__ import braces": "a = 1\nb = 2\nfrom __future__ import braces\n",
    "the same argument twice": "a = 1\ndef f(x, x):\n    pass\n",
    "await outside a function": "class A:\n    a = 1\n    b = [await c for c in a]\n    d = 2\n",
}


class PiecesTest(unittest.TestCase):
    def assert_same_tree(self, source, sizes=SIZES, must_split=SPLITS):
        expected = dump(whole(source))
        for size in sizes:
            tree = pp.parse_pieces(source, "<test>", size)
            if must_split:
                self.assertIsNotNone(tree, f"pieces of {size} lines were given up")
            if tree is not None:
                got = dump(tree)
                self.assertTrue(got == expected, f"pieces of {size} lines: " + difference(got, expected))
                self.assertTrue(same_tree(tree, whole(source)), f"same_tree, pieces of {size} lines")
            got = dump(pp.parse(source, "<test>", size))
            self.assertTrue(got == expected, f"parse, pieces of {size} lines: " + difference(got, expected))

    def test_each_boundary_rule_gives_the_whole_file_tree(self):
        for name, source in SOURCES.items():
            with self.subTest(name):
                self.assert_same_tree(source)

    def test_same_tree_sees_what_the_dumps_see(self):
        source = "class A:\n    x = 1.0\n    def m(self):\n        return [self.x, 'a']\ny = A\n"
        changes = {
            "a column": lambda t: setattr(t.body[0].body[1].body[0].value.elts[0], "col_offset", 99),
            "an end line": lambda t: setattr(t.body[0], "end_lineno", 99),
            "a name": lambda t: setattr(t.body[1].value, "id", "B"),
            "a constant": lambda t: setattr(t.body[0].body[0].value, "value", 2.0),
            "the sign of a zero": lambda t: setattr(t.body[0].body[0].value, "value", -0.0),
            "the type of a constant": lambda t: setattr(t.body[0].body[0].value, "value", 1),
            "a statement fewer": lambda t: t.body[0].body.pop(),
            "a statement more": lambda t: t.body.append(t.body[1]),
            "another kind of node": lambda t: t.body.__setitem__(1, ast.Pass(lineno=5, col_offset=0, end_lineno=5,
                                                                              end_col_offset=5)),
        }
        self.assertTrue(same_tree(whole(source), whole(source)))
        for name, change in changes.items():
            with self.subTest(name):
                tree = whole(source.replace("1.0", "0.0") if "zero" in name else source)
                reference = whole(source.replace("1.0", "0.0") if "zero" in name else source)
                change(tree)
                self.assertNotEqual(dump(tree), dump(reference))
                self.assertFalse(same_tree(tree, reference))
                self.assertFalse(same_tree(reference, tree))

    def test_the_same_from_bytes_with_lf_crlf_and_cr_line_ends(self):
        for name, source in SOURCES.items():
            for ending in ("\n", "\r\n", "\r"):
                with self.subTest(name, ending=ending):
                    text = source.replace("\n", ending)
                    self.assert_same_tree(text, sizes=(1, 3))
                    self.assert_same_tree(text.encode("utf-8"), sizes=(1, 3))

    def test_a_bom_and_a_coding_line_are_read_as_compile_reads_them(self):
        self.assert_same_tree(b"\xef\xbb\xbf" + NON_ASCII.encode("utf-8"))
        self.assert_same_tree(LATIN_1.encode("latin-1"))
        self.assertEqual(whole(LATIN_1.encode("latin-1")).body[0].value.value,"\xe9\xe8")
        # undecodable bytes and an unknown coding are compile's to report
        for data in (b"a = 1\nb = '\xff'\n", b"# coding: no-such-codec\na = 1\n"):
            with self.subTest(data):
                self.assertIsNone(pp.parse_pieces(data, "<test>", 1))
                self.assertIsNotNone(error_facts(whole, data))
                self.assertEqual(error_facts(pp.parse, data, "<test>", 1), error_facts(whole, data))

    def test_sources_with_nothing_to_cut(self):
        for source in ("", "\n", ONLY_COMMENTS, "x = 1", "x = 1\n", "class A: pass\n", '"""Only a docstring."""\n'):
            with self.subTest(source):
                self.assert_same_tree(source)

    def test_a_backslash_line_before_a_statement_is_given_up_not_guessed(self):
        # "\\\n" on a line of its own joins the next line, so the statement after it starts a row early: valid
        # Python that no cut may separate. The piece before it ends in the backslash and does not parse.
        whole(BACKSLASH_LINES)
        self.assert_same_tree(BACKSLASH_LINES, must_split=False)
        if SPLITS:
            self.assertIsNone(pp.parse_pieces(BACKSLASH_LINES, "<test>", 1))

    def test_the_class_ends_where_its_last_token_ends(self):
        # `x = 1;` ends at the 1 and its class at the semicolon: the end comes from the parser, not from the body
        source = "class A:\n    a = 1\n    b = 2\n    x = 1;\ny = A\n"
        self.assertNotEqual(whole(source).body[0].end_col_offset, whole(source).body[0].body[-1].end_col_offset)
        self.assert_same_tree(source)

    @unittest.skipUnless(SPLITS, "sources are parsed whole before Python 3.12")
    def test_a_long_class_is_cut_inside_and_other_long_statements_are_not(self):
        method = "    def m{0}(self):\n        return f'{{self}} {0}'\n\n"
        source = "class Big:\n" + "".join(method.format(i) for i in range(40)) + "x = 1\n"
        parsed = []
        real_compile = compile
        with mock.patch.object(pp, "compile", lambda text, *a, **k: parsed.append(text) or real_compile(text, *a, **k),
                               create=True):
            tree = pp.parse_pieces(source, "<test>", 30)
        self.assertEqual(dump(tree), dump(whole(source)))
        self.assertEqual(len(parsed), 5, [p.lstrip("\n")[:40] for p in parsed])
        self.assertEqual([p.lstrip("\n").startswith(pp.DUMMY) for p in parsed], [False, True, True, True, False])
        # a function of the same length is one statement and stays one piece
        function = "def big():\n" + "".join(f"    a{i} = f'{{{i}}}'\n" for i in range(120)) + "x = 1\n"
        del parsed[:]
        with mock.patch.object(pp, "compile", lambda text, *a, **k: parsed.append(text) or real_compile(text, *a, **k),
                               create=True):
            tree = pp.parse_pieces(function, "<test>", 30)
        self.assertEqual(dump(tree), dump(whole(function)))
        self.assertEqual(len(parsed), 2)
        # Each piece of a class starts a new indentation stack, so the tokenizer's tab rule between two statements
        # of the body would go unchecked: a body not indented by the same characters throughout is never cut.
        # (A form feed is the one valid way to get there; tabs against spaces are an error, see ErrorsTest.)
        mixed = "class A:\n    a = 1\n\x0c    b = 2\n    c = 3\nd = 4\n"
        del parsed[:]
        with mock.patch.object(pp, "compile", lambda text, *a, **k: parsed.append(text) or real_compile(text, *a, **k),
                               create=True):
            tree = pp.parse_pieces(mixed, "<test>", 1)
        self.assertEqual(dump(tree), dump(whole(mixed)))
        self.assertEqual([p.lstrip("\n").startswith(pp.DUMMY) for p in parsed], [False, False])

    def test_a_short_source_is_one_piece(self):
        with mock.patch.object(pp, "_statements", side_effect=AssertionError("tokenized a source of one piece")):
            self.assert_same_tree(DECORATORS, sizes=(pp.PIECE_LINES,))


class ErrorsTest(unittest.TestCase):
    def test_a_source_with_an_error_raises_what_compile_raises(self):
        for name, source in ERRORS.items():
            for variant in (source, source.encode("utf-8"), "ok = 0\n" * 12 + source, source + "ok = 0\n" * 12):
                expected = error_facts(whole, variant, "some/file.py")
                if variant is source:
                    self.assertIsNotNone(expected, f"compile accepts {name!r}")
                if expected is None:
                    continue                # the lines added after it completed the statement
                with self.subTest(name, variant=variant):
                    for size in SIZES + (pp.PIECE_LINES,):
                        self.assertIsNone(pp.parse_pieces(variant, "some/file.py", size), f"pieces of {size} lines")
                        self.assertEqual(error_facts(pp.parse, variant, "some/file.py", size), expected)

    def test_what_the_compiler_rejects_is_rejected_in_the_same_words(self):
        for name, source in COMPILE_ERRORS.items():
            with self.subTest(name):
                expected = error_facts(compile, source, "f.py", "exec")
                self.assertIsNotNone(expected, "compile accepts this source")
                for size in (1, 2):
                    got = error_facts(lambda: compile(pp.parse(source, "f.py", size), "f.py", "exec"))
                    self.assertEqual(got, expected, f"pieces of {size} lines")
                if error_facts(whole, source) is None:      # the parser takes it: the pieces give its tree
                    self.assertEqual(dump(pp.parse(source, "f.py", 1)), dump(whole(source)))

    def test_barry_as_flufl_is_parsed_whole(self):
        # The one __future__ import that changes the grammar for the rest of the file. Which of `<>` and `!=` the
        # parser then takes depends on the release: 3.12.16, 3.13.13 and 3.14.7 read the import from the source
        # (_PyPegen_checked_future_import), 3.12.1 and 3.13.0 look at the compile flags only. So nothing is said
        # here about which one is right: the pieces give up, and `parse` gives what this interpreter's compile gives.
        sources = [f"from __future__ import barry_as_FLUFL\na = 1\nb = a {operator} 2\n" for operator in ("<>", "!=")]
        for source in sources:
            with self.subTest(source=source):
                self.assertIsNone(pp.parse_pieces(source, "<test>", 1))
                self.assertEqual(outcome(pp.parse, source, "<test>", 1), outcome(whole, source))
        self.assertEqual(sorted(error_facts(whole, source) is None for source in sources), [False, True])


class WarningsTest(unittest.TestCase):
    SOURCE = ('a = 1\n'
              'b = "\\d"\n'
              'class W:\n'
              '    c = 2\n'
              '    d = "\\w" + "\\s"\n'
              '    e = 3\n'
              'f = 1if a else 2\n')

    @staticmethod
    def recorded(call, *args):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call(*args)
        return sorted((w.lineno, w.category.__name__, str(w.message), w.filename) for w in caught)

    def test_the_parser_warns_once_with_the_files_name_and_line(self):
        expected = self.recorded(whole, self.SOURCE, "warned.py")
        self.assertEqual([line for line, _, _, _ in expected], [2, 5, 5, 7], expected)
        self.assertEqual({name for _, _, _, name in expected}, {"warned.py"})
        for size in SIZES:
            self.assertEqual(self.recorded(pp.parse, self.SOURCE, "warned.py", size), expected, f"{size} lines")

    def test_the_compiler_warns_once_too(self):
        # compiling a tree repeats nothing the parser said, and says what compiling the source says
        for source in ("a = 1\nb = a is 1\nclass C:\n    c = 2\n    d = c is 2\n    e = 3\n",
                       "a = 1\nassert (a, 'always true')\nb = 2\n",
                       "def f():\n    try:\n        pass\n    finally:\n        return 1\ng = 2\n",
                       self.SOURCE):
            expected = self.recorded(compile, source, "warned.py", "exec")
            for size in (1, 2):
                got = self.recorded(lambda: compile(pp.parse(source, "warned.py", size), "warned.py", "exec"))
                self.assertEqual(got, expected, f"pieces of {size} lines")
        self.assertTrue(self.recorded(compile, "a = 1\nb = a is 1\n", "warned.py", "exec"))

    def test_a_warning_turned_into_an_error_is_the_error_compile_raises(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            expected = error_facts(whole, self.SOURCE, "warned.py")
            self.assertIsNotNone(expected)
            for size in SIZES:
                self.assertEqual(error_facts(pp.parse, self.SOURCE, "warned.py", size), expected)


class NeededTest(unittest.TestCase):
    def test_the_interpreters_that_copy_the_source_at_every_field(self):
        for version, expected in (((3, 10, 12), False), ((3, 11, 9), False), ((3, 12, 0), True), ((3, 12, 16), True),
                                  ((3, 12, 99), True), ((3, 13, 0), True), ((3, 13, 15), True), ((3, 13, 16), False),
                                  ((3, 13, 17), False), ((3, 14, 0), True), ((3, 14, 7), True), ((3, 14, 8), False),
                                  ((3, 14, 10), False), ((3, 15, 0), False), ((3, 16, 1), False)):
            self.assertIs(pp.needed(version + ("final", 0), "cpython"), expected, version)
        self.assertIs(pp.needed((3, 12, 4, "final", 0), "pypy"), False)
        self.assertIs(pp.needed(), pp.needed(sys.version_info, sys.implementation.name))

    def test_parse_takes_pieces_where_they_are_needed_or_asked_for(self):
        source = "".join(f"a{i} = f'{{{i}}}'\n" for i in range(pp.PIECE_LINES + 40))
        expected = dump(whole(source))
        for needed, sizes, pieces in ((True, (), 1), (False, (), 0), (False, (7,), 1), (True, (7,), 1)):
            with self.subTest(needed=needed, sizes=sizes):
                with mock.patch.object(pp, "needed", return_value=needed), \
                        mock.patch.object(pp, "parse_pieces", wraps=pp.parse_pieces) as parse_pieces:
                    self.assertEqual(dump(pp.parse(source, "<test>", *sizes)), expected)
                size = sizes[0] if sizes else pp.PIECE_LINES
                self.assertEqual(parse_pieces.call_args_list, [mock.call(source, "<test>", size)] * pieces)


class RepositoryTest(unittest.TestCase):
    def test_every_python_file_gives_the_whole_file_tree(self):
        checked = cut = 0
        for path in python_files():
            if os.path.samefile(path, TRANSLATOR):
                continue                    # test_the_translator_tree_and_code_object_slow
            data = read(path)
            with self.subTest(os.path.relpath(path, ROOT)):
                try:
                    expected = whole(data, path)
                except (SyntaxError, ValueError):           # a stray file that is not Python: refused, not guessed
                    self.assertIsNone(pp.parse_pieces(data, path, 1))
                    continue
                # one statement a piece for every file; the size the loader uses where the loader can meet it
                long_tool = path.startswith(TOOLS + os.sep) and data.count(b"\n") >= pp.PIECE_LINES
                for size in (1, pp.PIECE_LINES) if long_tool else (1,):
                    tree = pp.parse_pieces(data, path, size)
                    if tree is None:        # allowed for the one documented reason
                        self.assertFalse(SPLITS and b"barry_as_FLUFL" not in data, f"pieces of {size} given up")
                        continue
                    if not same_tree(tree, expected):
                        self.fail(f"pieces of {size} lines: " + difference(dump(tree), dump(expected)))
                    cut += 1
                checked += 1
        self.assertGreater(checked, 1000, "the scan no longer finds the repository's Python files")
        if SPLITS:
            self.assertGreater(cut, checked)

    def test_the_code_object_equals_the_one_compile_makes_from_the_bytes(self):
        for rel in ("make_game.py", "tools/shader_ir.py", "tools/spirv_asm.py", "tools/wave64_halves.py",
                    "scripts/run_gpu_tests.py", "scripts/run_semantic_tests.py", "tests/semantic/spirv_mini_interp.py"):
            path = os.path.join(ROOT, *rel.split("/"))
            data = read(path)
            expected = compile(data, path, "exec", dont_inherit=True)
            for size in (1, 40, pp.PIECE_LINES):
                with self.subTest(rel, size=size):
                    code = compile(pp.parse(data, path, size), path, "exec", dont_inherit=True)
                    self.assertEqual(code, expected)
                    self.assertTrue(code_facts(code) == code_facts(expected), "positions or nested code differ")

    def test_the_translator_tree_and_code_object_slow(self):
        data = read(TRANSLATOR)
        expected = dump(whole(data, TRANSLATOR))                            # seconds where the pieces are needed
        tree = pp.parse_pieces(data, TRANSLATOR)
        if SPLITS:
            self.assertIsNotNone(tree, "the pieces were given up")
        if tree is not None:
            got = dump(tree)
            self.assertTrue(got == expected, difference(got, expected))
            del got
        del expected, tree
        reference = compile(data, TRANSLATOR, "exec", dont_inherit=True)    # and seconds again
        code = support.PiecewiseLoader("rdna2spirv", TRANSLATOR).source_to_code(data, TRANSLATOR)
        self.assertEqual(code, reference)
        self.assertTrue(code_facts(code) == code_facts(reference), "positions or nested code differ")


MODULE = '''"""A module of a throwaway tools folder."""
import os

ANSWER = 41


class Thing:
    def name(self, x):
        return f"{x}:{ANSWER + 1}"


def where():
    return os.path.abspath(__file__)
''' + "".join(f"\n\ndef f{i}(x):\n    return f'{{x}}{i}'\n" for i in range(pp.PIECE_LINES // 4 + 5))


class LoaderTest(unittest.TestCase):
    """support.py's ToolsFinder and PiecewiseLoader, on a folder standing in for tools/."""

    def setUp(self):
        td = tempfile.TemporaryDirectory(prefix="leonida pieces ")
        self.addCleanup(td.cleanup)
        self.tools = os.path.join(td.name, "tools")
        self.other = os.path.join(td.name, "other")
        os.mkdir(self.tools)
        os.mkdir(self.other)
        for p in (mock.patch.object(support, "TOOLS_DIR", self.tools),
                  mock.patch.object(sys, "path", [self.tools] + sys.path)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(importlib.invalidate_caches)
        self.addCleanup(lambda: [sys.path_importer_cache.pop(d, None) for d in (self.tools, self.other)])

    def write(self, folder, name, text):
        path = os.path.join(folder, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(text.encode("utf-8"))
        importlib.invalidate_caches()
        return path

    def test_a_module_of_the_tools_folder_gets_the_loader_and_the_same_code(self):
        self.assertGreater(MODULE.count("\n"), pp.PIECE_LINES)          # long enough to be cut
        path = self.write(self.tools, "leonida_pieces_probe.py", MODULE)
        spec = support.ToolsFinder.find_spec("leonida_pieces_probe")
        self.assertIs(type(spec.loader), support.PiecewiseLoader)
        self.assertTrue(os.path.samefile(spec.origin, path))
        with mock.patch.object(pp, "parse_pieces", wraps=pp.parse_pieces) as used:
            code = spec.loader.get_code("leonida_pieces_probe")
        self.assertEqual(used.call_count, 1)
        reference = compile(MODULE.encode("utf-8"), spec.origin, "exec", dont_inherit=True)
        self.assertEqual(code, reference)
        self.assertTrue(code_facts(code) == code_facts(reference))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.Thing().name("a"), "a:42")
        self.assertEqual(module.f3(7), "73")
        self.assertTrue(os.path.samefile(module.where(), path))
        self.assertEqual(module.__doc__, "A module of a throwaway tools folder.")

    def test_names_it_leaves_to_the_usual_finders(self):
        self.write(self.tools, "leonida_pieces_shadowed.py", "WHERE = 'tools'\n")
        self.write(self.other, "leonida_pieces_shadowed.py", "WHERE = 'other'\n")
        self.write(self.other, "leonida_pieces_elsewhere.py", "X = 1\n")
        self.write(self.tools, os.path.join("leonida_pieces_package", "__init__.py"), "X = 1\n")
        self.write(self.tools, os.path.join("leonida_pieces_package", "inner.py"), "X = 1\n")
        with mock.patch.object(sys, "path", [self.other] + sys.path):
            # found first in another folder of sys.path: that file is the module, as without the finder
            self.assertIsNone(support.ToolsFinder.find_spec("leonida_pieces_shadowed"))
            self.assertIsNone(support.ToolsFinder.find_spec("leonida_pieces_elsewhere"))
        self.assertIsNone(support.ToolsFinder.find_spec("leonida_pieces_missing"))
        self.assertIsNone(support.ToolsFinder.find_spec("leonida_pieces_package"))
        self.assertIsNone(support.ToolsFinder.find_spec("leonida_pieces_package.inner",
                                                        [os.path.join(self.tools, "leonida_pieces_package")]))
        self.assertIsNone(support.ToolsFinder.find_spec("os"))
        self.assertIsNotNone(support.ToolsFinder.find_spec("leonida_pieces_shadowed"))

    def test_an_error_in_a_module_is_the_one_the_usual_loader_raises(self):
        for name, source in (("syntax", MODULE + "\nbroken = (\n"), ("compiler", MODULE + "\nreturn 1\n")):
            self.write(self.tools, f"leonida_pieces_bad_{name}.py", source)
            spec = support.ToolsFinder.find_spec(f"leonida_pieces_bad_{name}")
            self.assertIs(type(spec.loader), support.PiecewiseLoader)
            usual = importlib.machinery.SourceFileLoader(f"leonida_pieces_bad_{name}", spec.origin)
            expected = error_facts(usual.get_code, f"leonida_pieces_bad_{name}")
            self.assertIsNotNone(expected)
            self.assertEqual(error_facts(spec.loader.get_code, f"leonida_pieces_bad_{name}"), expected)

    def test_a_failure_in_the_pieces_falls_back_to_the_usual_compile(self):
        path = self.write(self.tools, "leonida_pieces_fallback.py", MODULE)
        reference = compile(MODULE.encode("utf-8"), path, "exec", dont_inherit=True)
        loader = support.PiecewiseLoader("leonida_pieces_fallback", path)
        for failure in (mock.patch.object(pp, "parse_pieces", return_value=None),
                        mock.patch.object(pp, "parse_pieces", side_effect=RuntimeError("anything")),
                        mock.patch.object(pp, "parse_pieces", return_value=ast.Module(     # a tree compile refuses
                            body=[ast.Expr(value=ast.Constant(value=1))], type_ignores=[]))):
            with failure:
                self.assertEqual(loader.get_code("leonida_pieces_fallback"), reference)

    def test_a_valid_pyc_is_used_without_parsing(self):
        path = self.write(self.tools, "leonida_pieces_cached.py", MODULE)
        py_compile.compile(path, doraise=True)
        spec = support.ToolsFinder.find_spec("leonida_pieces_cached")
        with mock.patch.object(support.PiecewiseLoader, "source_to_code", side_effect=AssertionError("parsed")):
            code = spec.loader.get_code("leonida_pieces_cached")
        self.assertEqual(code, compile(MODULE.encode("utf-8"), spec.origin, "exec", dont_inherit=True))

    def test_support_installs_the_finder_only_where_the_tokenizer_is_slow(self):
        translator = sys.modules["rdna2spirv"]
        self.assertTrue(os.path.samefile(translator.__file__, TRANSLATOR))
        if pp.needed():
            at = sys.meta_path.index(support.ToolsFinder)
            self.assertIs(sys.meta_path[at + 1], importlib.machinery.PathFinder)
            self.assertIs(type(translator.__spec__.loader), support.PiecewiseLoader)
        else:
            self.assertNotIn(support.ToolsFinder, sys.meta_path)
            self.assertIs(type(translator.__spec__.loader), importlib.machinery.SourceFileLoader)


if __name__ == "__main__":
    unittest.main()
