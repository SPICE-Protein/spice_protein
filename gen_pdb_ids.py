#!/usr/bin/env python3
"""Generate the PDB ID list for SPICE data download.

Queries the RCSB Search API for entries annotated with an environment
(pH AND temperature in crystal-growth conditions), paginates every result,
and writes a newline-separated ID list (lowercase, one per line).

NOTE on ionic strength:
  The RCSB Search API has *no* cleanly searchable numeric ionic-strength
  field (it only lives in free-text `exptl_crystal_grow.pdbx_details`, or
  in the non-searchable `pdbx_nmr_exptl_sample_conditions`). Ionic strength
  is therefore filtered later, in the cleaning stage, by parsing each
  downloaded mmCIF. This script only pre-filters the API-searchable part.

Usage:
    python3 gen_pdb_ids.py [out.txt]        # default: pdb_ids.txt
"""

import json
import sys
import time
import urllib.error
import urllib.request

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    tqdm = None
    HAS_TQDM = False

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"

# ---------------------------------------------------------------------------
# Criteria — tunable. Default: pH AND temperature both present in
# exptl_crystal_grow, AND resolution <= RES_MAX_ANGSTROM.
#
# NOTE: `refine.ls_d_res_high` is NOT searchable (HTTP 400); the correct
# searchable resolution field is `rcsb_entry_info.resolution_combined`
# (covers both X-ray and EM). `rcsb_entry_info.r_free` and
# `rcsb_cluster_member.*` are ALSO not searchable — R-free and sequence
# redundancy must be handled post-download (rfree column / MMseqs2 clustering).
# Ionic strength has no searchable numeric field — filtered later by `has_env`.
# ---------------------------------------------------------------------------
# Max resolution (Å) for accepted structures. None disables the filter.
RES_MAX_ANGSTROM = 2.5

# ---- additional quality filters for a high-quality pre-train set ----
# Protein-only: exclude DNA/RNA-containing entries.
PROTEIN_ONLY = True
# Single polymer entity: exclude protein complexes / multi-entity assemblies.
SINGLE_ENTITY = True
# Single chain instance (STRICTER than SINGLE_ENTITY — excludes homo-oligomers).
# Pick ONE of SINGLE_ENTITY / SINGLE_CHAIN (leave the other False).
SINGLE_CHAIN = False
# Polymer length bounds (residues) — MD-friendly range. None disables either bound.
LEN_MIN = 40
LEN_MAX = 400

QUERY = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "exptl_crystal_grow.pH",
                    "operator": "exists",
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "exptl_crystal_grow.temp",
                    "operator": "exists",
                },
            },
        ],
    },
    "return_type": "entry",
    "request_options": {
        "paginate": {"start": 0, "rows": 10000},
        "results_content_type": ["experimental"],
    },
}

if RES_MAX_ANGSTROM is not None:
    QUERY["query"]["nodes"].append(
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal",
                "value": RES_MAX_ANGSTROM,
            },
        }
    )

if PROTEIN_ONLY:
    QUERY["query"]["nodes"].append(
        {"type": "terminal", "service": "text",
         "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_nucleic_acid",
                        "operator": "equals", "value": 0}}
    )
if SINGLE_ENTITY:
    QUERY["query"]["nodes"].append(
        {"type": "terminal", "service": "text",
         "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count",
                        "operator": "equals", "value": 1}}
    )
if SINGLE_CHAIN:
    QUERY["query"]["nodes"].append(
        {"type": "terminal", "service": "text",
         "parameters": {"attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                        "operator": "equals", "value": 1}}
    )
if LEN_MIN is not None:
    QUERY["query"]["nodes"].append(
        {"type": "terminal", "service": "text",
         "parameters": {"attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                        "operator": "greater_or_equal", "value": LEN_MIN}}
    )
if LEN_MAX is not None:
    QUERY["query"]["nodes"].append(
        {"type": "terminal", "service": "text",
         "parameters": {"attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                        "operator": "less_or_equal", "value": LEN_MAX}}
    )

PAGE_ROWS = 10000
REQUEST_DELAY = 0.2  # seconds between pages; be polite


def post(query: dict) -> dict:
    req = urllib.request.Request(
        SEARCH_URL,
        data=json.dumps(query).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "pdb_ids.txt"

    ids: set[str] = set()
    total: int | None = None
    start = 0
    pbar = None

    while True:
        q = json.loads(json.dumps(QUERY))
        q["request_options"]["paginate"] = {"start": start, "rows": PAGE_ROWS}
        try:
            data = post(q)
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code} at start={start}: {e.read()[:200]!r}", file=sys.stderr)
            return 1
        except Exception as e:  # transient network errors -> retry once
            print(f"error at start={start}: {e}; retrying in 5s", file=sys.stderr)
            time.sleep(5)
            continue

        if total is None:
            total = int(data.get("total_count", 0))
            print(f"total_count (API): {total}")
            if HAS_TQDM:
                pbar = tqdm(total=total, unit="id", desc="Collecting PDB IDs")

        results = data.get("result_set", [])
        for r in results:
            ids.add(r["identifier"])
        got = len(results)
        if pbar is not None:
            pbar.update(got)
            pbar.set_postfix(pages=start // PAGE_ROWS + 1, refresh=False)
        if got == 0 or start + got >= total:
            break
        start += got
        time.sleep(REQUEST_DELAY)

    if pbar is not None:
        pbar.close()

    ordered = sorted(ids, key=str.lower)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ordered) + "\n")

    print(f"wrote {len(ordered)} IDs -> {out_path}")
    if total is not None and len(ids) < total:
        print(
            f"WARNING: only {len(ids)}/{total} collected — some pages failed",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
