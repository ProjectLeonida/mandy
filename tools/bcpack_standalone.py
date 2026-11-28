"""Exact BC1-BC7 reconstruction using the vendored format constants.

SPDX-License-Identifier: MIT
Ported from Xbox2Windows d049001; see third_party/x2w-bcpack for attribution.
This module does not load, train, fit or override predictor dictionaries.
"""
import json
from pathlib import Path
import struct

TABLES = json.loads(Path(__file__).with_name('bcpack_standalone_tables.json').read_text(encoding='utf-8'))


def zig(v):
    return (v >> 1) ^ -(v & 1)


def inter(a, b, bits=4):
    return sum(((a >> k & 1) | ((b >> k & 1) << 1)) << (2*k) for k in range(bits))


def predict(i, r, history, dictionary, zero):
    return zero if r == 0 else history[(i-r) & 255] if r <= i else dictionary[r-i-1]


def colour(ep, ix):
    codes = (0, 2, 3, 1)
    c = [(ep[3*e] << 11) | (ep[3*e+1] << 5) | ep[3*e+2] for e in range(2)]
    return struct.pack('<HHI', *c, sum(codes[v & 3] << (2*p) for p, v in enumerate(ix)))


def alpha(ep, ix):
    codes = (0, 2, 3, 4, 5, 6, 7, 1)
    v = (ep[0] ^ 128) | ((ep[1] ^ 128) << 8)
    v |= sum(codes[x & 7] << (16+3*p) for p, x in enumerate(ix))
    return v.to_bytes(8, 'little')


def quads(rank, symbols, predictor):
    result = [0]*16
    for q, symbol in enumerate(symbols):
        base = (q >> 1)*8 + (q & 1)*2
        for k, offset in enumerate((0, 1, 4, 5)):
            p = base+offset
            result[p] = (predictor[p] + rank[symbol][k]) & 3
    return result


def pairs(rank, symbols, predictor):
    return [(predictor[2*k+h] + rank[s & 63][h]) & 7 for k, s in enumerate(symbols) for h in range(2)]


def classic(fmt, records):
    t = TABLES
    dictionary = t[{1:'BC1_FULL_DICTIONARY', 2:'BC2_FULL_DICTIONARY', 3:'BC3_DICTIONARY',
                    4:'BC4_DICTIONARY', 5:'BC5_DICTIONARY'}[fmt]]
    if fmt == 2:
        zero = [[0]*16, [0]*6, [0]*16]
    elif fmt == 3:
        zero = [[0]*2, [0]*6, [0]*16, [0]*16]
        dictionary = [[ep[:2], ep[2:], ix[:16], ix[16:]] for ep, ix in dictionary]
    else:
        zero = [[0]*(6 if fmt == 1 else 2 if fmt == 4 else 4), [0]*(32 if fmt == 5 else 16)]
    past = [zero]*256
    out = bytearray()
    for i, s in enumerate(records):
        paired = fmt not in (1, 4)
        d, xr = (inter(*s[:2]), inter(*s[2:4])) if paired else s[:2]
        ep = predict(i, (xr+zig(d)) & 255, past, dictionary, zero)
        ix = predict(i, xr, past, dictionary, zero)
        if fmt == 1:
            e = [(ep[0][k]+zig(s[2+k])) & ((1 << b)-1) for k,b in enumerate((5,6,5,5,6,5))]
            x = quads(t['BC1_POSITION_RANK'],s[8:12],ix[1])
            state = [e,x]
            out += colour(e,x)
        elif fmt == 2:
            a = [(ep[0][k]+zig(s[4+k])) & 15 for k in range(16)]
            e = [(ep[1][3*e+c]+zig(s[20+2*c+e])) & ((1 << (6 if c == 1 else 5))-1)
                 for e in range(2) for c in range(3)]
            x = quads(t['BC2_POSITION_RANK'],s[26:30],ix[2])
            state = [a,e,x]
            out += sum((v ^ 8) << (4*p) for p,v in enumerate(a)).to_bytes(8,'little') + colour(e,x)
        elif fmt == 3:
            a = [(ep[0][k]+zig(s[4+k])) & 255 for k in range(2)]
            e = [(ep[1][3*e+c]+zig(s[14+2*c+e])) & ((1 << (6 if c == 1 else 5))-1)
                 for e in range(2) for c in range(3)]
            ax = pairs(t['RANK_PAIR_BC3_ALPHA'],s[6:14],ix[2])
            cx = quads(t['RANK_QUAD_BC3_COLOUR'],s[20:24],ix[3])
            state = [a,e,ax,cx]
            out += alpha(a,ax)+colour(e,cx)
        elif fmt == 4:
            e = [(ep[0][k]+zig(s[2+k])) & 255 for k in range(2)]
            x = pairs(t['RANK_PAIR_BC4'],s[4:12],ix[1])
            state = [e,x]
            out += alpha(e,x)
        else:
            e = [(ep[0][k]+zig(s[4+k])) & 255 for k in range(4)]
            x = pairs(t['RANK_PAIR_BC5'],s[8:24],ix[1])
            state = [e,x]
            out += alpha(e[:2],x[:16])+alpha(e[2:],x[16:])
        past[i & 255] = state
    return bytes(out)


def pair_residual(v, row):
    halves = [sum((v >> (2*k+h) & 1) << k for k in range(4)) for h in range(2)]
    return [zig(x) for x in (halves[::-1] if row < 3 else halves)]


def bc6(records):
    t = TABLES
    zero = [[[0]*3 for _ in range(4)], [0]*16]
    past = [zero]*256
    out = bytearray()
    codes = (0,1,2,6,10,14,18,22,26,30,3,7,11,15)
    for i,s in enumerate(records):
        mode = s[30]+1
        if not 1 <= mode <= 14:
            raise ValueError('invalid BC6H mode')
        part = s[31] & 31
        two = mode <= 10
        xr = inter(*s[2:4])
        ep = predict(i,(xr+zig(inter(*s[:2]))) & 255,past,t['BC6_DICTIONARY'],zero)
        ix = predict(i,xr,past,t['BC6_DICTIONARY'],zero)
        e = [[0]*3 for _ in range(4)]
        def add(base,delta,shift):
            return (base + (delta << shift)) & (65535 << shift) & 65535
        for c in range(3):
            s0,sc = 16-t['BC6_ENDPOINT_BITS'][mode][0],16-t['BC6_ENDPOINT_BITS'][mode][c+1]
            e[0][c] = add(ep[0][0][c],zig(inter(*s[4+2*c:6+2*c],bits=8)),s0)
            e[1][c] = add(ep[0][1][c],zig(inter(*s[10+2*c:12+2*c],bits=8)),sc)
            if two:
                e[2][c] = add(ep[0][2][c],zig(s[16+2*c]),sc)
                e[3][c] = add(ep[0][3][c],zig(s[17+2*c]),sc)
            else:
                e[2][c] = ((e[0][c] >> s0) << sc) & 65535
                e[3][c] = e[1][c]
        bits = 3 if two else 4
        x = [add(ix[1][2*k+h],dv,16-bits) for k in range(8)
             for h,dv in enumerate(pair_residual(s[22+k],k//2))]
        value = 0
        header = 82 if two else 65
        for at, (field, bit) in enumerate(t['BC6_WIRE'][mode-1][:header]):
            if field == 0: v = codes[mode-1]
            elif field == 1: v = part
            else:
                k,c = divmod(field-2,3)
                shift = 16-t['BC6_ENDPOINT_BITS'][mode][0 if k == 0 else c+1]
                v = (e[k][c] ^ 32768) >> shift
            value |= ((v >> bit) & 1) << at
        at = header
        anchor = t['ANCHOR_2'][part] if two else 0
        for p,v in enumerate(x):
            width = bits-int(p in (0,anchor))
            v = (v >> (16-bits)) + (1 << (bits-1))
            value |= (v & ((1 << width)-1)) << at
            at += width
        out += value.to_bytes(16,'little')
        past[i & 255] = [e,x]
    return bytes(out)


def bc7(records):
    t = TABLES
    zero = [0,[[0]*8 for _ in range(3)],[[0]*16 for _ in range(2)]]
    past = [zero]*256
    out = bytearray()
    for i,s in enumerate(records):
        code = ((s[31] & 15) << 5) | (s[30] & 31)
        mode,part,rotation,select,valid = t['BC7_CONTROL_TABLE'][code]
        if not valid:
            raise ValueError('invalid BC7 control')
        subsets = t['SUBSETS'][mode]
        xr = inter(*s[2:4])
        ep = predict(i,(xr+zig(inter(*s[:2]))) & 255,past,t['BC7_DICTIONARY'],zero)
        ix = predict(i,xr,past,t['BC7_DICTIONARY'],zero)
        prec = list(t['CHANNEL_BITS'][mode])
        pbits = mode in (0,1,3,6,7)
        if rotation: prec[3],prec[rotation-1] = prec[rotation-1],prec[3]
        elif pbits: prec[0] += 1
        e = [[0]*8 for _ in range(3)]
        common = min(subsets,ep[0])
        for ch,p in enumerate(prec):
            if not p: continue
            for row in range(subsets):
                if row == 0: lane = ch+2
                elif row == 1: lane = ch+6
                elif ch == 0: lane = 10
                elif ch == 1: lane = 5
                elif ch == 2: lane = 9
                else: lane = -1
                if subsets == 3 and ch == 3: lane = -1
                for endpoint in range(2):
                    col = 2*ch+endpoint
                    delta = zig(s[2*lane+endpoint]) if lane >= 0 else 0
                    base = ep[1][row if row < common else 0][col] >> (8-p)
                    e[row][col] = ((base+delta) & ((1 << p)-1)) << (8-p)
        x = [[0]*16 for _ in range(2)]
        for set,bits in enumerate(t['INDEX_BITS'][mode]):
            if not bits: continue
            for row,lane in enumerate(t['INDEX_LANES'][set]):
                for half in range(2):
                    for h,delta in enumerate(pair_residual(s[2*lane+half],row)):
                        p = 4*row+2*half+h
                        x[set][p] = (((ix[2][set][p] >> (4-bits))+delta) & ((1 << bits)-1)) << (4-bits)
        value = at = 0
        def put(v,n):
            nonlocal value,at
            value |= (v & ((1 << n)-1)) << at
            at += n
        put(1 << mode,mode+1)
        put(part,t['PARTITION_BITS'][mode])
        if mode in (4,5): put(rotation,2)
        if mode == 4: put(select,1)
        order = [0,1,2,3]
        if rotation: order[3],order[rotation-1] = order[rotation-1],order[3]
        for ch in order:
            p = prec[ch]
            if not p: continue
            drop = int(ch == 0 and pbits)
            for endpoint in range(2*subsets):
                put((e[endpoint//2][2*ch+endpoint%2] >> (8-p)) >> drop,p-drop)
        if mode == 1:
            for row in range(subsets): put(e[row][0] >> (8-prec[0]),1)
        elif pbits:
            for endpoint in range(2*subsets): put(e[endpoint//2][endpoint%2] >> (8-prec[0]),1)
        anchors = {0}
        if subsets == 2: anchors.add(t['ANCHOR_2'][part])
        elif subsets == 3: anchors.update((t['ANCHOR_3A'][part],t['ANCHOR_3B'][part]))
        for set in ((1,0) if mode == 4 else (0,1) if mode == 5 else (0,)):
            bits = t['INDEX_BITS'][mode][set]
            for p,v in enumerate(x[set]): put(v >> (4-bits),bits-int(p in anchors))
        out += value.to_bytes(16,'little')
        past[i & 255] = [subsets,e,x]
    return bytes(out)


def rebuild(lanes, codec, blocks):
    per = 1 if codec in (1,4) else 2
    records = [[lanes[lane][i*per+h] for lane in range(len(lanes)) for h in range(per)] for i in range(blocks)]
    return classic(codec,records) if codec <= 5 else bc6(records) if codec == 6 else bc7(records)
