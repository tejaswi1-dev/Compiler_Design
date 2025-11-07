#!/usr/bin/env python3
"""
Region-Based Register Allocator with CFG, labels, goto, and if-goto
Standalone educational prototype
Requires: networkx, matplotlib
Usage: python region_alloc_cfg.py program.ir
"""

import sys
import re
from collections import defaultdict, namedtuple
import networkx as nx
import matplotlib.pyplot as plt

NUM_REGS = 4
REGISTERS = [f"r{i}" for i in range(1, NUM_REGS + 1)]

# ----- Parser -----
Instr = namedtuple("Instr", ["type", "dest", "op1", "op", "op2", "label", "raw"])

def parse_line(line):
    line = line.strip()
    if not line:
        return None
    # label
    if line.endswith(":"):
        lbl = line[:-1].strip()
        return Instr("label", None, None, None, None, lbl, line)
    # goto
    m = re.match(r'goto\s+(\w+)$', line)
    if m:
        return Instr("goto", None, None, None, None, m.group(1), line)
    # if X goto LABEL  (nonzero test)
    m = re.match(r'if\s+(\w+)\s+goto\s+(\w+)$', line)
    if m:
        return Instr("if", None, m.group(1), None, None, m.group(2), line)
    # print
    m = re.match(r'print\s+(\w+)$', line)
    if m:
        return Instr("print", None, m.group(1), None, None, None, line)
    # assignment: x = y  OR x = y op z
    m = re.match(r'(\w+)\s*=\s*(.+)$', line)
    if m:
        dest = m.group(1)
        rhs = m.group(2).strip()
        # try binary op
        m2 = re.match(r'(\w+|\d+)\s*([\+\-\*\/])\s*(\w+|\d+)$', rhs)
        if m2:
            op1, op, op2 = m2.group(1), m2.group(2), m2.group(3)
            return Instr("assign", dest, op1, op, op2, None, line)
        # simple copy / const
        return Instr("assign", dest, rhs, None, None, None, line)
    raise ValueError("Cannot parse line: " + line)

# ----- Read file and parse -----
if len(sys.argv) < 2:
    print("Usage: python region_alloc_cfg.py <program.ir>")
    sys.exit(1)

with open(sys.argv[1]) as f:
    raw_lines = [l.rstrip() for l in f if l.strip() and not l.strip().startswith("#")]

instrs = [parse_line(l) for l in raw_lines]

# Map labels -> instr index
label_to_idx = {}
for i, ins in enumerate(instrs):
    if ins.type == "label":
        label_to_idx[ins.label] = i

# ----- Build leaders -> basic blocks (classical algorithm) -----
leaders = set()
leaders.add(0)
for i, ins in enumerate(instrs):
    if ins.type == "label":
        leaders.add(i)
    elif ins.type == "goto":
        tgt = label_to_idx.get(ins.label)
        if tgt is not None:
            leaders.add(tgt)
        if i + 1 < len(instrs):
            leaders.add(i + 1)
    elif ins.type == "if":
        tgt = label_to_idx.get(ins.label)
        if tgt is not None:
            leaders.add(tgt)
        if i + 1 < len(instrs):
            leaders.add(i + 1)

leaders = sorted(leaders)
# form blocks by leader ranges
blocks = []
leader_to_bid = {}
for i, L in enumerate(leaders):
    start = L
    end = leaders[i+1] if i+1 < len(leaders) else len(instrs)
    block_instrs = []
    for j in range(start, end):
        # skip label-only if it's the first of block (we keep it for reference)
        block_instrs.append((j, instrs[j]))
    bid = len(blocks)
    blocks.append(block_instrs)
    leader_to_bid[L] = bid

# map instruction index -> block id
idx_to_bid = {}
for bid, blk in enumerate(blocks):
    for idx, _ in blk:
        idx_to_bid[idx] = bid

# ----- Build CFG: edges between blocks -----
cfg = defaultdict(set)
for bid, blk in enumerate(blocks):
    last_idx, last_ins = blk[-1]
    if last_ins.type == "goto":
        tgt_idx = label_to_idx[last_ins.label]
        cfg[bid].add(idx_to_bid[tgt_idx])
    elif last_ins.type == "if":
        tgt_idx = label_to_idx[last_ins.label]
        cfg[bid].add(idx_to_bid[tgt_idx])
        # fall-through if exists
        if bid + 1 < len(blocks):
            cfg[bid].add(bid + 1)
    else:
        # fall-through if exists
        if bid + 1 < len(blocks):
            cfg[bid].add(bid + 1)

# For visualization / sanity
print(f"Blocks formed: {len(blocks)}")
for bid, blk in enumerate(blocks):
    print(f" B{bid}: instr idxs {[i for i,_ in blk]}")
print("CFG edges:")
for k, v in cfg.items():
    print(f" B{k} -> {sorted(v)}")

# ----- Liveness analysis on CFG (backwards) -----
def defs_uses_block(blk):
    defs, uses = set(), set()
    for idx, ins in blk:
        if ins.type == "assign":
            # dest is a def
            defs.add(ins.dest)
            # sources are uses if they are names (alphabetic)
            if ins.op is None:
                if isinstance(ins.op1, str) and ins.op1.isalpha():
                    uses.add(ins.op1)
            else:
                if isinstance(ins.op1, str) and ins.op1.isalpha():
                    uses.add(ins.op1)
                if isinstance(ins.op2, str) and ins.op2.isalpha():
                    uses.add(ins.op2)
        elif ins.type == "if":
            if isinstance(ins.op1, str) and ins.op1.isalpha():
                uses.add(ins.op1)
        elif ins.type == "print":
            if isinstance(ins.op1, str) and ins.op1.isalpha():
                uses.add(ins.op1)
        # labels/goto have no uses/defs
    return defs, uses

nb = len(blocks)
live_in = [set() for _ in range(nb)]
live_out = [set() for _ in range(nb)]
changed = True
while changed:
    changed = False
    for b in reversed(range(nb)):
        defs, uses = defs_uses_block(blocks[b])
        new_out = set()
        for s in cfg.get(b, []):
            new_out |= live_in[s]
        new_in = uses | (new_out - defs)
        if new_in != live_in[b] or new_out != live_out[b]:
            live_in[b], live_out[b] = new_in, new_out
            changed = True

print("\nLiveness (per block):")
for b in range(nb):
    print(f" B{b}: IN={sorted(live_in[b])} OUT={sorted(live_out[b])}")

# ----- Region partitioning: SCCs of the CFG (loops become regions) -----
Gcfg = nx.DiGraph()
Gcfg.add_nodes_from(range(nb))
for u, vs in cfg.items():
    for v in vs:
        Gcfg.add_edge(u, v)

sccs = list(nx.strongly_connected_components(Gcfg))
# order sccs by smallest block id for readability
sccs = sorted(sccs, key=lambda s: min(s))
regions = [sorted(list(s)) for s in sccs]
print("\nRegions (SCCs):", regions)

# ----- Interference graph per region & coloring -----
def build_interference_for_region(region_blocks):
    G = nx.Graph()
    # process instructions in region in reverse program order
    # collect blocks' instructions indices in reverse
    region_instr_indices = []
    for b in sorted(region_blocks):
        region_instr_indices += [idx for idx, _ in blocks[b]]
    # iterate backward
    live = set()
    for idx in reversed(region_instr_indices):
        ins = instrs[idx]
        # ensure nodes exist for variables seen
        if ins.type == "assign":
            G.add_node(ins.dest)
            if ins.op is None:
                if isinstance(ins.op1, str) and ins.op1.isalpha():
                    G.add_node(ins.op1)
            else:
                if isinstance(ins.op1, str) and ins.op1.isalpha():
                    G.add_node(ins.op1)
                if isinstance(ins.op2, str) and ins.op2.isalpha():
                    G.add_node(ins.op2)
            # defs interfere with everything currently live
            for d in [ins.dest]:
                for l in live:
                    if l != d:
                        G.add_edge(d, l)
            # update live: kill defs, add uses
            # remove def
            if ins.dest in live:
                live.remove(ins.dest)
            # add uses
            if ins.op is None:
                if isinstance(ins.op1, str) and ins.op1.isalpha():
                    live.add(ins.op1)
            else:
                if isinstance(ins.op1, str) and ins.op1.isalpha():
                    live.add(ins.op1)
                if isinstance(ins.op2, str) and ins.op2.isalpha():
                    live.add(ins.op2)
        elif ins.type == "if":
            if isinstance(ins.op1, str) and ins.op1.isalpha():
                G.add_node(ins.op1); live.add(ins.op1)
        elif ins.type == "print":
            if isinstance(ins.op1, str) and ins.op1.isalpha():
                G.add_node(ins.op1); live.add(ins.op1)
        # labels/gotos don't change live
    return G

def greedy_color(G, regs):
    coloring = {}
    # nodes sorted by degree descending
    order = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
    for n in order:
        used = set(coloring.get(nb) for nb in G[n] if nb in coloring)
        found = None
        for r in regs:
            if r not in used:
                found = r; break
        coloring[n] = found  # None means spilled
    return coloring

alloc_by_region = {}
globally_spilled = set()
for ridx, region in enumerate(regions):
    G = build_interference_for_region(region)
    coloring = greedy_color(G, REGISTERS)
    alloc_by_region[ridx] = coloring
    for v, c in coloring.items():
        if c is None:
            globally_spilled.add(v)
    # visualize
    if len(G.nodes) > 0:
        pos = nx.spring_layout(G, seed=42)
        colors = ['lightgreen' if coloring.get(n) else 'salmon' for n in G.nodes()]
        plt.figure(figsize=(4,3))
        nx.draw(G, pos, with_labels=True, node_color=colors, node_size=800)
        plt.title(f"Region {ridx} Interference Graph")
        plt.tight_layout()
        plt.savefig(f"region_{ridx}_graph.png", dpi=150)
        plt.close()

# ----- Produce final allocated code (conceptual spills as memory slots) -----
allocated_lines = []
# rule for reading mapping: if a var is globally spilled, use [spill_x] when reading/writing
def fmt_read(var):
    if var is None:
        return "?"
    if isinstance(var, str) and var.isalpha() and var in globally_spilled:
        return f"[spill_{var}]"
    return str(var)

def fmt_dest(var):
    if var in globally_spilled:
        return f"[spill_{var}]"
    # try to find region coloring - pick first region that defines it
    for ridx, color in alloc_by_region.items():
        if var in color and color[var] is not None:
            return color[var]
    # If not allocated into reg, show spilled
    return f"[spill_{var}]"

for bid, blk in enumerate(blocks):
    allocated_lines.append(f"// Block B{bid}:")
    for idx, ins in blk:
        if ins.type == "label":
            allocated_lines.append(f"{ins.label}:")
        elif ins.type == "goto":
            allocated_lines.append(f"    goto {ins.label}")
        elif ins.type == "if":
            src = fmt_read(ins.op1)
            allocated_lines.append(f"    if {src} != 0 goto {ins.label}")
        elif ins.type == "print":
            src = fmt_read(ins.op1)
            allocated_lines.append(f"    print {src}")
        elif ins.type == "assign":
            dest_loc = fmt_dest(ins.dest)
            if ins.op is None:
                src = fmt_read(ins.op1)
                allocated_lines.append(f"    {dest_loc} = {src}")
            else:
                s1 = fmt_read(ins.op1)
                s2 = fmt_read(ins.op2)
                allocated_lines.append(f"    {dest_loc} = {s1} {ins.op} {s2}")

# ----- Stats & Outputs -----
print("\n=== Allocation Summary ===")
for ridx, alloc in alloc_by_region.items():
    print(f"\nRegion {ridx} blocks {regions[ridx]}:")
    for v, r in sorted(alloc.items()):
        print(f"  {v:8} -> {r or 'SPILLED'}")
print("\nGlobally spilled vars:", sorted(globally_spilled))

print("\n=== Allocated (conceptual) code ===")
for l in allocated_lines:
    print(l)

# ----- Register pressure per block (live count) plot -----
pressure = [len(live_in[b]) + len(live_out[b]) for b in range(nb)]
plt.figure(figsize=(6,3))
plt.plot(range(nb), pressure, marker='o')
plt.xlabel("Block id")
plt.ylabel("approx register pressure (in+out)")
plt.title("Register pressure per block (in+out)")
plt.grid(True)
plt.tight_layout()
plt.savefig("register_pressure.png", dpi=150)
plt.close()
print("\nSaved: region_*.png and register_pressure.png")
