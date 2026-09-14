#!/usr/bin/env python3
"""Headless summary of the fracture library - no window, no engine.

    python3 tools_fracture_report.py [--seed N] [--json OUT.json]

Prints a table of chunk counts and size ranges per structure archetype, so the
fracture output can be sanity-checked at a glance without launching the game.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game import fracture as F  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--json", metavar="PATH", help="also dump raw stats as JSON")
    args = ap.parse_args()

    specs = F.default_specs()
    rows = []
    for spec in specs:
        r = F.fracture(spec, args.seed)
        vols = r.volumes()
        ars = sorted(c.aspect_ratio for c in r.chunks)
        rows.append({
            "spec": r.spec_name,
            "kind": r.spec_kind,
            "blocks": len(spec.blocks),
            "budget": spec.max_chunks,
            "chunks": len(r),
            "src_volume": r.source_volume,
            "chunk_volume": r.chunk_volume,
            "vol_err": r.volume_error,
            "vmin": vols[0],
            "vmed": statistics.median(vols),
            "vmax": vols[-1],
            "vratio": vols[-1] / vols[0],
            "ar_med": statistics.median(ars),
            "ar_max": ars[-1],
            "bands": F.size_bands(r, 3),
            "total_mass_t": r.total_mass / 1000.0,
            "gen_ms": r.generation_seconds * 1000.0,
        })

    hdr = (
        f"{'structure':<14}{'kind':<9}{'blk':>4}{'bud':>5}{'chunks':>7}"
        f"{'src vol':>10}{'chunk vol':>11}{'vol err':>10}"
        f"{'min v':>9}{'med v':>9}{'max v':>10}{'max/min':>9}"
        f"{'ar med':>8}{'ar max':>8}{'S/M/L bands':>16}{'mass kt':>9}{'gen ms':>8}"
    )
    print()
    print("TUMBLE FRACTURE LIBRARY - per-archetype summary   (seed %d)" % args.seed)
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        bands = "/".join(str(b) for b in r["bands"])
        print(
            f"{r['spec']:<14}{r['kind']:<9}{r['blocks']:>4}{r['budget']:>5}"
            f"{r['chunks']:>7}{r['src_volume']:>10.1f}{r['chunk_volume']:>11.1f}"
            f"{r['vol_err']:>10.1e}{r['vmin']:>9.3f}{r['vmed']:>9.3f}"
            f"{r['vmax']:>10.2f}{r['vratio']:>9.1f}{r['ar_med']:>8.2f}"
            f"{r['ar_max']:>8.2f}{bands:>16}{r['total_mass_t'] / 1000.0:>9.2f}"
            f"{r['gen_ms']:>8.1f}"
        )
    print("=" * len(hdr))
    print(
        "vol err = |sum(chunk volume) - source volume| / source volume\n"
        "S/M/L bands = chunk counts in three equal log10-volume bands\n"
        "ar = aspect ratio of the chunk's local bounding box (1.0 = cube)"
    )

    total_ms = sum(r["gen_ms"] for r in rows)
    print(f"\nall {len(rows)} structures pre-generated in {total_ms:.1f} ms "
          f"({sum(r['chunks'] for r in rows)} chunks total)")

    print("\nseed sensitivity (chunk count / max-volume chunk, arch spec):")
    arch = F.arch_spec()
    for s in (1, 2, 3, 4, 5):
        r = F.fracture(arch, s)
        print(f"  seed {s}: {len(r):>4} chunks, largest {r.volumes()[-1]:7.2f} m^3, "
              f"vol err {r.volume_error:.1e}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2, sort_keys=True)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
