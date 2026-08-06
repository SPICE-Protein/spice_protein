#!/usr/bin/env python3
"""Convert downloaded mmCIF biological assemblies to Parquet, keeping only
the fragments SPICE needs, then delete the raw files.

Input : data/raw/{id}-assembly1.cif.gz   (output of download_parallel.py)
Output: data/parquet/entries_shard_*.parquet  (one row per structure)
        data/parquet/atoms_shard_*.parquet    (one row per atom, long format)

Raw files are removed after successful conversion, so re-running this
script naturally resumes: only files still present are re-processed.

Schema
------
entries (per structure):
    pdb_id, method, resolution, ph, temperature,
    ionic_strength_m, has_ionic, seq, n_residues, has_env

atoms (per atom):
    pdb_id, chain_id, res_seq, res_name, atom_name, element,
    x, y, z, occupancy, b_factor, is_ca

Run with the `spice` conda env:
    conda activate spice
    python convert_to_parquet.py --raw data/raw --out data/parquet
"""

import argparse
import gzip
import os
import re
import sys

import gemmi
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# 1-letter amino-acid map (used when _entity_poly sequence is unavailable)
# ---------------------------------------------------------------------------
ONELETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z", "XAA": "X",
    "UNK": "X", "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C",
    "CYM": "C", "GLH": "E", "ASH": "D", "LYN": "K", "TYM": "Y",
}

# ---------------------------------------------------------------------------
# Ionic-strength extraction from free-text `pdbx_details`
# (heuristic — phrase variants vary wildly; tune as needed)
# ---------------------------------------------------------------------------
HAS_SALT_RE = re.compile(
    r"(NaCl|KCl|LiCl|MgCl2|CaCl2|NH4Cl|ammonium sulfate|ammonium sulphate|"
    r"ionic strength|salt|Na2SO4|K2SO4)",
    re.I,
)
IONIC_PATTERNS = [
    re.compile(r"ionic strength\s+of?\s*([\d.]+)\s*M", re.I),
    re.compile(
        r"([\d.]+)\s*M\s*(NaCl|KCl|LiCl|MgCl2|MgCl2|CaCl2|NH4Cl|"
        r"ammonium sulfate|ammonium sulphate|Na2SO4|K2SO4|"
        r"sodium chloride|potassium chloride)",
        re.I,
    ),
    re.compile(
        r"(sodium chloride|NaCl|KCl|MgCl2)\s+([\d.]+)\s*M", re.I,
    ),
]


def parse_ionic(details: str | None) -> tuple[float | None, bool]:
    if not details:
        return None, False
    has = bool(HAS_SALT_RE.search(details))
    for pat in IONIC_PATTERNS:
        m = pat.search(details)
        if m:
            try:
                return float(m.group(1)), has
            except ValueError:
                pass
    return None, has


def _as_float(block: gemmi.cif.Block, tag: str) -> float | None:
    try:
        raw = block.find_value(tag)
    except Exception:
        return None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _as_str(block: gemmi.cif.Block, tag: str) -> str | None:
    try:
        raw = block.find_value(tag)
    except Exception:
        return None
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


def _parse_block(path: str) -> tuple[dict | None, list[dict]]:
    """Parse one cif.gz file -> (entry_row | None, atom_rows)."""
    try:
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                doc = gemmi.cif.read_string(f.read())
        else:
            doc = gemmi.cif.read_file(path)
        block = doc.sole_block()
    except Exception as e:
        print(f"  ! parse failed {os.path.basename(path)}: {e}", file=sys.stderr)
        return None, []

    pdb_id = os.path.basename(path).split(".")[0].upper()

    # ---- environment / structure-level metadata ----
    ph = _as_float(block, "_exptl_crystal_grow.pH")
    temp = _as_float(block, "_exptl_crystal_grow.temp")
    details = _as_str(block, "_exptl_crystal_grow.pdbx_details")
    ionic_m, has_ionic = parse_ionic(details)
    method = _as_str(block, "_exptl.method")
    resolution = (
        _as_float(block, "_refine.ls_d_res_high")
        or _as_float(block, "_em_3d_reconstruction.resolution")
        or _as_float(block, "_reflns.d_resolution_high")
    )

    # ---- atom_site loop ----
    loop = block.find_mmcif_category("_atom_site")
    tags = list(loop.tags)
    def col(name: str) -> int:
        return tags.index(name) if name in tags else -1

    i_g = col("_atom_site.group_PDB")
    i_an = col("_atom_site.auth_atom_id")
    i_an2 = col("_atom_site.label_atom_id")
    i_rn = col("_atom_site.auth_comp_id")
    i_rn2 = col("_atom_site.label_comp_id")
    i_chain = col("_atom_site.auth_asym_id")
    i_chain2 = col("_atom_site.label_asym_id")
    i_res = col("_atom_site.auth_seq_id")
    i_res2 = col("_atom_site.label_seq_id")
    i_x = col("_atom_site.Cartn_x")
    i_y = col("_atom_site.Cartn_y")
    i_z = col("_atom_site.Cartn_z")
    i_occ = col("_atom_site.occupancy")
    i_b = col("_atom_site.B_iso_or_equiv")
    i_el = col("_atom_site.type_symbol")

    atom_rows = []
    residues_seen = set()
    for row in loop:
        if i_g >= 0 and row[i_g] != "ATOM":
            continue  # keep only polymer atoms
        if i_x < 0 or row[i_x] == "?" or row[i_x] == ".":
            continue
        an = row[i_an] if i_an >= 0 else row[i_an2]
        rn = row[i_rn] if i_rn >= 0 else row[i_rn2]
        chain = row[i_chain] if i_chain >= 0 else row[i_chain2]
        res = row[i_res] if i_res >= 0 else row[i_res2]
        try:
            res_i = int(float(res))
        except (TypeError, ValueError):
            res_i = 0
        occ = _num(row, i_occ)
        b = _num(row, i_b)
        el = row[i_el] if i_el >= 0 else ""
        is_ca = an in ("CA", "Cα")
        residues_seen.add((chain, res_i))
        atom_rows.append({
            "pdb_id": pdb_id, "chain_id": chain, "res_seq": res_i,
            "res_name": rn, "atom_name": an, "element": el,
            "x": _num(row, i_x), "y": _num(row, i_y), "z": _num(row, i_z),
            "occupancy": occ, "b_factor": b, "is_ca": is_ca,
        })

    if not atom_rows:
        return None, []

    # ---- sequence: prefer _entity_poly, else rebuild from CA residues ----
    seq = _as_str(block, "_entity_poly.pdbx_seq_one_letter_code")
    if seq:
        seq = re.sub(r"\s+", "", seq)
        seq = "".join(ch for ch in seq if ch.isalpha())
    else:
        letters = []
        for chain, res_i in sorted(residues_seen):
            name = next(
                (r["res_name"] for r in atom_rows
                 if r["chain_id"] == chain and r["res_seq"] == res_i and r["is_ca"]),
                None,
            )
            letters.append(ONELETTER.get(str(name).upper(), "X"))
        seq = "".join(letters)

    n_res = len(residues_seen)
    has_env = bool(ph is not None and temp is not None and has_ionic)

    entry = {
        "pdb_id": pdb_id, "method": method, "resolution": resolution,
        "ph": ph, "temperature": temp,
        "ionic_strength_m": ionic_m, "has_ionic": has_ionic,
        "seq": seq, "n_residues": n_res, "has_env": has_env,
    }
    return entry, atom_rows


def _num(row, i) -> float | None:
    if i < 0 or row[i] in ("?", "."):
        return None
    try:
        return float(row[i])
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Parquet schemas
# ---------------------------------------------------------------------------
ENTRY_SCHEMA = pa.schema([
    ("pdb_id", pa.string()), ("method", pa.string()),
    ("resolution", pa.float64()), ("ph", pa.float64()),
    ("temperature", pa.float64()), ("ionic_strength_m", pa.float64()),
    ("has_ionic", pa.bool_()), ("seq", pa.string()),
    ("n_residues", pa.int32()), ("has_env", pa.bool_()),
])

ATOM_SCHEMA = pa.schema([
    ("pdb_id", pa.string()), ("chain_id", pa.string()),
    ("res_seq", pa.int32()), ("res_name", pa.string()),
    ("atom_name", pa.string()), ("element", pa.string()),
    ("x", pa.float32()), ("y", pa.float32()), ("z", pa.float32()),
    ("occupancy", pa.float32()), ("b_factor", pa.float32()),
    ("is_ca", pa.bool_()),
])


def _table(rows: list[dict], schema: pa.Schema) -> pa.Table:
    cols = {}
    for f in schema:
        name = f.name
        vals = [r.get(name) for r in rows]
        cols[name] = pa.array(vals, type=f.type)
    return pa.Table.from_pydict(cols, schema=schema)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", help="dir with {id}-assembly1.cif.gz")
    ap.add_argument("--out", default="data/parquet")
    ap.add_argument("--chunk", type=int, default=5000,
                    help="structures per output shard file")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = sorted(
        f for f in os.listdir(args.raw)
        if f.endswith(".cif.gz") and not f.startswith("_")
    )
    if not files:
        print(f"no *.cif.gz in {args.raw}; nothing to convert")
        return 0
    print(f"converting {len(files)} files -> {args.out}")

    shard = 0
    n_entries = n_atoms = 0
    entry_buf, atom_buf = [], []
    for i, fn in enumerate(files, 1):
        entry, atoms = _parse_block(os.path.join(args.raw, fn))
        if entry is None:
            # delete unparseable junk so we don't loop forever on reruns
            os.remove(os.path.join(args.raw, fn))
            continue
        entry_buf.append(entry)
        atom_buf.extend(atoms)
        n_entries += 1
        n_atoms += len(atoms)

        if len(entry_buf) >= args.chunk:
            shard += 1
            pq.write_table(_table(entry_buf, ENTRY_SCHEMA),
                           os.path.join(args.out, f"entries_shard_{shard:04d}.parquet"))
            pq.write_table(_table(atom_buf, ATOM_SCHEMA),
                           os.path.join(args.out, f"atoms_shard_{shard:04d}.parquet"))
            # raw files removed only after their data is safely on disk
            for _e, _a, fn_ in zip(entry_buf, atom_buf, files[i - len(entry_buf):i]):
                os.remove(os.path.join(args.raw, fn_))
            entry_buf, atom_buf = [], []

        if i % 2000 == 0:
            print(f"  [{i}/{len(files)}] entries={n_entries} atoms={n_atoms}")

    if entry_buf:
        shard += 1
        pq.write_table(_table(entry_buf, ENTRY_SCHEMA),
                       os.path.join(args.out, f"entries_shard_{shard:04d}.parquet"))
        pq.write_table(_table(atom_buf, ATOM_SCHEMA),
                       os.path.join(args.out, f"atoms_shard_{shard:04d}.parquet"))
        for fn_ in files[len(files) - len(entry_buf):]:
            os.remove(os.path.join(args.raw, fn_))

    print(f"done: {n_entries} entries, {n_atoms} atoms -> {shard} shards")
    return 0


if __name__ == "__main__":
    sys.exit(main())
