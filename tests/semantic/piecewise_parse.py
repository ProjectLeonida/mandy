"""Parse a Python source in pieces and join them into the tree a whole-file parse gives (standard library only).

    tree = piecewise_parse.parse(source, filename)      # equal to ast.parse(source, filename), positions included

Why: from CPython 3.12.0 (PEP 701) the tokenizer copies the rest of its input at every replacement field of an
f-string: `update_fstring_expr` in Parser/tokenizer.c (in Parser/lexer/lexer.c from 3.13, as
`_PyLexer_update_fstring_expr` and then `_PyLexer_update_ftstring_expr`) does `strlen(tok->cur)`, `PyMem_Malloc`
and `strncpy`. `import`, `compile` and `ast.parse` hand the module over as one string, so "the rest" is the rest of
the file and the time grows with the square of its size: tools/rdna2spirv.py (1.2 MB, 4,759 f-strings with about
12,700 fields) takes 7 s to parse and 0.1 s to turn into code. A file run as a script is read line by line and is
not affected. CPython 3.14.8 and 3.15.0 fixed it (gh-155525); 3.12, which CI runs, only gets security fixes now and
keeps it, and so does every 3.13 up to 3.13.15. `needed()` says whether the running interpreter is one of those.

What a field costs depends on how much source follows it, so the same statements are cheap in a short string:

* `tokenize` reads line by line (linear). It gives the depth of every logical line (INDENT/DEDENT) and its first
  token. A statement starts at a logical line that does not begin with `elif`, `else`, `except` or `finally` and
  does not follow a decorator line.
* Consecutive top-level statements are gathered into pieces of about PIECE_LINES lines. Each piece is parsed with
  one "\\n" in front for every line before it, so every node keeps its line and column.
* A longer top-level `class` is cut between the statements of its body. The first piece, with the header, gives the
  real ClassDef. Each later piece is parsed under `class _:` written on the line before it, its statements are
  appended to the real body, and the class ends where the last piece's `class _:` ends (both end at the last token
  of the body).

A piece starts where the whole-file tokenizer starts a logical line with the same indentation stack and ends where
it has closed every bracket and string, so the tokens are the same, and no rule of the grammar looks past a
statement except for the clauses and decorators kept together above. Whatever does not fit is not guessed at:
`parse_pieces` returns None for a source it cannot split, a piece that does not parse, or a piece in which a row
`tokenize` gave is not where a statement starts, and the caller compiles the whole source the usual way, with the
usual exception. A SyntaxWarning the parser gives for a line of a piece (an invalid escape, say) is given once,
with the file's name and line, as in a whole-file parse; only a source that then fails in a later piece gets it a
second time, from the whole-file parse that reports the error.

tests/test_piecewise_parse.py compares the trees, every field and position, for every Python file of the
repository (`ast.dump(..., include_attributes=True)` for the hand-written cases and the translator), and the code
objects for some.
"""
import ast
import importlib.util
import io
import sys
import tokenize

# tools/rdna2spirv.py on Python 3.14.7: 0.89 s in pieces of 50 or 100 lines, 0.90 s at 300, 1.10 s at 1,000,
# 1.85 s at 5,000 and 7.0 s whole. Below 300 the copying is already gone and smaller pieces only add calls.
PIECE_LINES = 300
DUMMY = "class _:\n"

_CONTINUES = frozenset(("elif", "else", "except", "finally"))
_BLANK = frozenset((tokenize.NL, tokenize.COMMENT, tokenize.ENDMARKER))


class _Whole(Exception):
    """This source is to be parsed whole."""


# What tokenizing or parsing a piece can raise for a source that a whole-file parse may still accept, or reports
# in its own words: all of them mean "parse it whole".
_GIVE_UP = (_Whole, SyntaxError, ValueError, RecursionError, MemoryError, OverflowError, tokenize.TokenError)


def needed(version=None, implementation=None):
    """True on the interpreters whose tokenizer copies the rest of the source at every f-string field: CPython
    3.12.0 up to 3.14.7 (see the module docstring). Elsewhere a whole-file parse is as fast and simpler."""
    version = tuple((sys.version_info if version is None else version)[:3])
    implementation = sys.implementation.name if implementation is None else implementation
    return implementation == "cpython" and (3, 12) <= version < (3, 14, 8)


def parse(source, filename="<unknown>", piece_lines=PIECE_LINES):
    """The module tree of `source` (str, or the bytes of a file): what `ast.parse(source, filename)` returns, and
    the same exception for a source with an error."""
    tree = parse_pieces(source, filename, piece_lines)
    if tree is None:
        tree = compile(source, filename, "exec", ast.PyCF_ONLY_AST, dont_inherit=True)
    return tree


def parse_pieces(source, filename="<unknown>", piece_lines=PIECE_LINES):
    """The same tree built from pieces of about `piece_lines` lines, or None when the source has to be parsed whole
    (see the module docstring). Nothing is raised for a source with an error: the whole-file parse reports it."""
    if sys.version_info < (3, 12):
        # Before 3.12 `tokenize` is not the parser's tokenizer (it does not apply the tab rule, for one), so the
        # statements it finds prove nothing about what the parser accepts; and there is nothing to gain there.
        return None
    try:
        if not isinstance(source, str):
            source = importlib.util.decode_source(source)       # as import does: coding line, BOM, newlines
        if "\r" in source:                                      # the tokenizer reads "\r\n" and "\r" as "\n"
            source = source.replace("\r\n", "\n").replace("\r", "\n")
        # A NUL is an error. `from __future__ import barry_as_FLUFL` changes what the parser accepts after it.
        if "\0" in source or "barry_as_FLUFL" in source:
            return None
        rows = io.StringIO(source).readlines()                  # split at "\n" only, as tokenize numbers them
        if len(rows) <= piece_lines:                            # one piece: nothing to look for
            return compile(source, filename, "exec", ast.PyCF_ONLY_AST, dont_inherit=True)

        def statements(first, end, header=""):
            """What rows first..end-1 parse to, `header` standing on the row before them."""
            text = "\n" * (first - 1 - bool(header)) + header + "".join(rows[first - 1:end - 1])
            return compile(text, filename, "exec", ast.PyCF_ONLY_AST, dont_inherit=True).body

        body = []
        for first, end, cuts, starts in _pieces(_statements(source), len(rows), piece_lines):
            found = statements(first, cuts[0] if cuts else end)
            body += found
            if cuts:
                if len(found) != 1 or type(found[0]) is not ast.ClassDef:
                    raise _Whole
                for at, stop in zip(cuts, cuts[1:] + [end]):
                    rest = statements(at, stop, DUMMY)
                    if len(rest) != 1 or type(rest[0]) is not ast.ClassDef:
                        raise _Whole
                    found[0].body += rest[0].body
                found[0].end_lineno, found[0].end_col_offset = rest[0].end_lineno, rest[0].end_col_offset
                found = found[0].body
            if not _rows(found).issuperset(starts):     # a row `tokenize` gave is not where a statement starts
                raise _Whole
        return ast.Module(body=body, type_ignores=[])
    except _GIVE_UP:
        return None


def _rows(nodes):
    """The rows where these statements start, decorators included. A statement after a `;` can start on a row of
    its own (`b` when `a = (` is closed by `); b = 2` on the next row), so there can be more rows than logical
    lines."""
    return {min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", ())]) for node in nodes}


def _statements(source):
    """[row, word, rows] for every top-level statement: the row it starts on, the word its own line starts with
    (the one after its decorators; None for a statement that must not be cut) and the rows where the statements
    one level down start."""
    tops = []
    depth = 0
    fresh = True            # the next token that is not blank starts a logical line
    before = None           # (depth, first token) of the logical line before
    indent = None           # what the statement one level down before this one is indented by
    for kind, word, (row, column), _, line in tokenize.generate_tokens(io.StringIO(source).readline):
        if kind == tokenize.INDENT:
            depth += 1
        elif kind == tokenize.DEDENT:
            depth -= 1
        elif kind == tokenize.NEWLINE:
            fresh = True
        elif fresh and kind not in _BLANK:
            fresh = False
            if depth < 2:
                decorated = before == (depth, "@")
                if not tops and (depth or decorated or word in _CONTINUES):
                    raise _Whole                        # indented, or a clause, before any statement
                if depth:
                    if not decorated and word not in _CONTINUES:
                        tops[-1][2].append(row)
                        if indent not in (None, line[:column]):
                            # Each piece of a class starts a new indentation stack, so the tokenizer's tab rule
                            # between two statements of its body would go unchecked: such a body is not cut.
                            tops[-1][1] = None
                        indent = line[:column]
                elif decorated:
                    tops[-1][1] = word
                elif word in _CONTINUES:
                    tops[-1][1] = None                  # several clauses: not a class
                else:
                    tops.append([row, word, []])
                    indent = None
            before = (depth, word)
    return tops


def _pieces(tops, last_row, piece_lines):
    """(first row, row after the last, cuts, starts) of every piece, in order. For a run of whole statements `cuts`
    is empty and `starts` are the rows they start on; for a class too long for one piece `cuts` are the rows where
    its later pieces start and `starts` the rows where the statements of its body start."""
    start, starts = 1, []                           # the first piece takes the lines above the first statement
    ends = [top[0] for top in tops[1:]] + [last_row + 1]
    for (row, word, inner), end in zip(tops, ends):
        cuts, at = [], row
        if word == "class" and end - row > piece_lines:
            for below in inner[1:]:                 # the first statement of the body stays with the header
                if below - at >= piece_lines:
                    cuts.append(below)
                    at = below
        if cuts:
            if row > start:
                yield start, row, [], starts
            yield row, end, cuts, inner
            start, starts = end, []
        else:
            starts.append(row)
            if end - start >= piece_lines:
                yield start, end, [], starts
                start, starts = end, []
    if start <= last_row:
        yield start, last_row + 1, [], starts
