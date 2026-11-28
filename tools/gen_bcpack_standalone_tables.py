"""Reproduce Python/Rust constants from the vendored MIT BCPack source tables."""
import argparse
import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

def tables():
    text = '\n'.join((ROOT / 'third_party/x2w-bcpack' / name).read_text(encoding='utf-8')
                     for name in ('bcpack_tables.h', 'bcpack_tables_bcn.inl', 'bcpack_tables_extra.inl', 'bcpack_decoder.h'))
    text = re.sub(r'//[^\n]*|/\*.*?\*/', '', text, flags=re.S)
    result = {}
    for match in re.finditer(r'inline constexpr\s+\w+\s+(\w+)\s*(?:\[[^\]]*\])+\s*=\s*(\{.*?\});', text, re.S):
        body = re.sub(r'\b(0x[0-9a-fA-F]+|[0-9]+)[uUlL]+\b', r'\1', match[2])
        result[match[1]] = ast.literal_eval(body.replace('{', '[').replace('}', ']'))
    return result

def rust_bc2(data):
    def value(v):
        return '[' + ', '.join(map(value, v)) + ']' if isinstance(v, list) else str(v)
    rank = data['BC2_POSITION_RANK']
    dictionary = data['BC2_FULL_DICTIONARY']
    return ('// Generated from third_party/x2w-bcpack; SPDX-License-Identifier: MIT\n'
            + '#[rustfmt::skip]\n'
            + 'pub(super) const RANK: [[u8; 4]; 256] = ' + value(rank) + ';\n'
            + '#[rustfmt::skip]\n'
            + 'pub(super) const DICTIONARY: [([u8; 16], [u8; 6], [u8; 16]); 255] = [\n'
            + ''.join('    (' + ', '.join(value(v) for v in row) + '),\n' for row in dictionary) + '];\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    data = tables()
    outputs = {
        ROOT / 'tools/bcpack_standalone_tables.json': json.dumps(data, separators=(',', ':')) + '\n',
        ROOT / 'd3d12/crates/leonida-formats/src/bcpack_bc2_tables.rs': rust_bc2(data),
    }
    for path, text in outputs.items():
        if args.check:
            if path.read_text(encoding='utf-8') != text:
                raise SystemExit('stale: ' + str(path.relative_to(ROOT)))
        else:
            path.write_text(text, encoding='utf-8', newline='\n')
    print('BCPack shared format constants: ' + ('verified' if args.check else 'generated'))
