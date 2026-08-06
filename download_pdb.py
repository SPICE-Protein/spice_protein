#!/usr/bin/env python3
"""One-shot async download + convert pipeline for SPICE PDB data.

Asynchronous download (aiohttp) of full-entry mmCIF files, parse the
fragments SPICE needs with gemmi, and write them as Parquet via polars
in sliding chunks (chunk-full -> write -> clear, so memory stays bounded).
Raw files are deleted after conversion. Resumable, live progress bar.

WHY full entry (not biological assembly): the RCSB `{id}-assembly1.cif.gz`
files do NOT carry the environment metadata (`_exptl_crystal_grow.pH/.temp`)
that SPICE needs; the full entry does.

Usage (Python 3.10+, run on Colab / server):
    pip install aiohttp aiofiles gemmi polars pyarrow tqdm
    python download_pdb.py pdb_ids.txt --out data --jobs 64 [--limit 100]

Outputs:
    data/raw/...                 (transient, deleted after conversion)
    data/parquet/entries_*.parquet   (one row per structure)
    data/parquet/atoms_*.parquet     (one row per atom, long format)
    data/parquet/.converted      (resume marker: already-processed IDs)
    data/parquet/_failures.txt   (download/parse failures)

Resume: already-processed IDs (in `.converted`) and files already present
in `data/raw` are skipped, so an interrupted run can be restarted safely.
"""

import argparse
import asyncio
import gzip
import os
import re
import sys
import time

import aiofiles
import aiohttp
import gemmi
import polars as pl

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BASE_URL = "https://files.wwpdb.org/download"  # 官方推荐域名
SUFFIX = ".cif.gz"
RETRIES = 3
MIN_VALID_SIZE = 100
TIMEOUT = aiohttp.ClientTimeout(total=90)

# Environment range validation (applied during local parse — PDB may hold junk).
PH_MIN, PH_MAX = 0.0, 14.0
TEMP_MIN, TEMP_MAX = 150.0, 400.0

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
# Ionic-strength heuristic over free-text `pdbx_details` (tunable)
# ---------------------------------------------------------------------------
HAS_SALT_RE = re.compile(
    r"(NaCl|KCl|LiCl|MgCl2|CaCl2|NH4Cl|ammonium sulfate|ammonium sulphate|"
    r"ionic strength|salt|Na2SO4|K2SO4)",
    re.I,
)
IONIC_PATTERNS = [
    re.compile(r"ionic strength\s+of?\s*([\d.]+)\s*M", re.I),
    re.compile(
        r"([\d.]+)\s*M\s*(NaCl|KCl|LiCl|MgCl2|CaCl2|NH4Cl|"
        r"ammonium sulfate|ammonium sulphate|Na2SO4|K2SO4|"
        r"sodium chloride|potassium chloride)",
        re.I,
    ),
    re.compile(r"(sodium chloride|NaCl|KCl|MgCl2)\s+([\d.]+)\s*M", re.I),
]


def parse_ionic(details):
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


def _as_float(block, tag):
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


def _as_str(block, tag):
    try:
        raw = block.find_value(tag)
    except Exception:
        return None
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


def _num(row, i):
    if i < 0 or row[i] in ("?", "."):
        return None
    try:
        return float(row[i])
    except (TypeError, ValueError):
        return None


def parse_block(path: str):
    """Parse one cif.gz file -> (entry_row | None, atom_rows)."""
    try:
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                doc = gemmi.cif.read_string(f.read())
        else:
            doc = gemmi.cif.read_file(path)
        block = doc.sole_block()
    except Exception:
        return None, []

    pdb_id = os.path.basename(path).split(".")[0].upper()

    ph = _as_float(block, "_exptl_crystal_grow.pH")
    temp = _as_float(block, "_exptl_crystal_grow.temp")
    rfree = _as_float(block, "_refine.R_FREE")
    details = _as_str(block, "_exptl_crystal_grow.pdbx_details")
    ionic_m, has_ionic = parse_ionic(details)
    method = _as_str(block, "_exptl.method")
    resolution = (
        _as_float(block, "_refine.ls_d_res_high")
        or _as_float(block, "_em_3d_reconstruction.resolution")
        or _as_float(block, "_reflns.d_resolution_high")
    )
    # Validate environment values are physically plausible (PDB may hold junk).
    ph_ok = ph is not None and PH_MIN <= ph <= PH_MAX
    temp_ok = temp is not None and TEMP_MIN <= temp <= TEMP_MAX

    loop = block.find_mmcif_category("_atom_site")
    tags = list(loop.tags)

    def col(name):
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
            continue
        if i_x < 0 or row[i_x] in ("?", "."):
            continue
        an = row[i_an] if i_an >= 0 else row[i_an2]
        rn = row[i_rn] if i_rn >= 0 else row[i_rn2]
        chain = row[i_chain] if i_chain >= 0 else row[i_chain2]
        res = row[i_res] if i_res >= 0 else row[i_res2]
        try:
            res_i = int(float(res))
        except (TypeError, ValueError):
            res_i = 0
        residues_seen.add((chain, res_i))
        atom_rows.append({
            "pdb_id": pdb_id, "chain_id": chain, "res_seq": res_i,
            "res_name": rn, "atom_name": an,
            "element": row[i_el] if i_el >= 0 else "",
            "x": _num(row, i_x), "y": _num(row, i_y), "z": _num(row, i_z),
            "occupancy": _num(row, i_occ), "b_factor": _num(row, i_b),
            "is_ca": an in ("CA", "Cα"),
        })

    if not atom_rows:
        return None, []

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

    entry = {
        "pdb_id": pdb_id, "method": method, "resolution": resolution,
        "ph": ph, "temperature": temp, "rfree": rfree,
        "ionic_strength_m": ionic_m, "has_ionic": has_ionic,
        "seq": seq, "n_residues": len(residues_seen),
        "has_env": bool(ph_ok and temp_ok and has_ionic),
    }
    return entry, atom_rows


# ---------------------------------------------------------------------------
# Parquet schemas (polars)
# ---------------------------------------------------------------------------
ENTRY_SCHEMA = {
    "pdb_id": pl.String, "method": pl.String, "resolution": pl.Float64,
    "ph": pl.Float64, "temperature": pl.Float64, "rfree": pl.Float64,
    "ionic_strength_m": pl.Float64,
    "has_ionic": pl.Boolean, "seq": pl.String, "n_residues": pl.Int32,
    "has_env": pl.Boolean,
}

ATOM_SCHEMA = {
    "pdb_id": pl.String, "chain_id": pl.String, "res_seq": pl.Int32,
    "res_name": pl.String, "atom_name": pl.String, "element": pl.String,
    "x": pl.Float32, "y": pl.Float32, "z": pl.Float32,
    "occupancy": pl.Float32, "b_factor": pl.Float32, "is_ca": pl.Boolean,
}


def write_sliding_chunk(entries, atoms, ids, parq_dir, shard_no, raw_dir, conv_marker):
    """Write one chunk to Parquet (polars), then delete the raw files."""
    shard_no += 1
    pl.DataFrame(entries, schema=ENTRY_SCHEMA).write_parquet(
        os.path.join(parq_dir, f"entries_shard_{shard_no:04d}.parquet")
    )
    pl.DataFrame(atoms, schema=ATOM_SCHEMA).write_parquet(
        os.path.join(parq_dir, f"atoms_shard_{shard_no:04d}.parquet")
    )
    with open(conv_marker, "a", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")
    for ident in ids:
        src = os.path.join(raw_dir, ident.lower() + SUFFIX)
        if os.path.exists(src):
            os.remove(src)
    return shard_no


# ---------------------------------------------------------------------------
# Async download + parse
# ---------------------------------------------------------------------------
async def fetch_one(session, ident, raw_dir):
    """Download one file (skip if present & valid). Returns (status, path)."""
    dest = os.path.join(raw_dir, ident.lower() + SUFFIX)
    if os.path.exists(dest) and os.path.getsize(dest) > MIN_VALID_SIZE:
        return "exists", dest

    url = f"{BASE_URL}/{ident.lower()}{SUFFIX}"
    for attempt in range(RETRIES):
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    text = await resp.text(errors="ignore")
                    if "xml" in text[:20].lower():
                        return "error-response", dest
                    return f"HTTP {resp.status}", dest
                data = await resp.read()
                if len(data) < MIN_VALID_SIZE:
                    return "too-small", dest
                tmp = dest + ".part"
                async with aiofiles.open(tmp, "wb") as f:
                    await f.write(data)
                os.replace(tmp, dest)
                return "ok", dest
        except Exception as e:
            if attempt < RETRIES - 1:
                await asyncio.sleep(2 ** attempt)  # 指数退避：1, 2, 4 秒
            else:
                return f"{type(e).__name__}: {e}", dest
    return "unknown", dest


async def process_one(session, sem, ident, raw_dir):
    """Download then parse (gemmi offloaded to a thread). Returns (ident, entry, atoms, status)."""
    async with sem:
        status, path = await fetch_one(session, ident, raw_dir)
    if status not in ("ok", "exists"):
        return ident, None, None, f"fail:{status}"
    entry, atoms = await asyncio.to_thread(parse_block, path)
    if entry is None:
        return ident, None, None, "parse_fail"
    return ident, entry, atoms, status


async def run(ids, raw_dir, parq_dir, chunk, jobs):
    sem = asyncio.Semaphore(jobs)
    connector = aiohttp.TCPConnector(limit=jobs * 2, limit_per_host=jobs)
    fail_log = os.path.join(parq_dir, "_failures.txt")
    conv_marker = os.path.join(parq_dir, ".converted")

    existing = [f for f in os.listdir(parq_dir) if f.startswith("entries_shard_")]
    shard_no = len(existing)

    pbar = tqdm(total=len(ids), unit="entry", disable=tqdm is None)
    ok = skipped = failed = parse_failed = 0
    buf_entries, buf_atoms, buf_ids = [], [], []
    t0 = time.time()

    async with aiohttp.ClientSession(connector=connector, timeout=TIMEOUT) as session:
        tasks = [asyncio.create_task(process_one(session, sem, i, raw_dir)) for i in ids]
        for coro in asyncio.as_completed(tasks):
            ident, entry, atoms, status = await coro
            key = status.split(":")[0]
            if entry is not None:
                buf_entries.append(entry)
                buf_atoms.extend(atoms)
                buf_ids.append(ident)
                if key == "exists":
                    skipped += 1
                else:
                    ok += 1
                if len(buf_entries) >= chunk:
                    shard_no = write_sliding_chunk(
                        buf_entries, buf_atoms, buf_ids, parq_dir, shard_no,
                        raw_dir, conv_marker,
                    )
                    buf_entries, buf_atoms, buf_ids = [], [], []
            elif key == "parse_fail":
                parse_failed += 1
                fail_dir = os.path.join(raw_dir, "_parse_failed")
                os.makedirs(fail_dir, exist_ok=True)
                src = os.path.join(raw_dir, ident.lower() + SUFFIX)
                if os.path.exists(src):
                    os.replace(src, os.path.join(fail_dir, ident.lower() + SUFFIX))
            else:
                failed += 1
                with open(fail_log, "a", encoding="utf-8") as f:
                    f.write(f"{ident}\t{status}\n")
            if pbar is not None:
                elapsed = time.time() - t0
                pbar.set_postfix(
                    ok=ok, skip=skipped, fail=failed,
                    rate=f"{(ok+skipped+failed)/elapsed if elapsed else 0:.1f}/s",
                    refresh=False,
                )
                pbar.update(1)

    if buf_entries:
        shard_no = write_sliding_chunk(
            buf_entries, buf_atoms, buf_ids, parq_dir, shard_no, raw_dir, conv_marker,
        )

    if pbar is not None:
        pbar.close()
    return ok, skipped, failed, parse_failed, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("idlist")
    ap.add_argument("--out", default="data")
    ap.add_argument("--jobs", type=int, default=32, help="concurrent downloads")
    ap.add_argument("--chunk", type=int, default=5000, help="structures per parquet shard")
    ap.add_argument("--limit", type=int, default=0, help="only first N IDs")
    args = ap.parse_args()

    raw_dir = os.path.join(args.out, "raw")
    parq_dir = os.path.join(args.out, "parquet")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(parq_dir, exist_ok=True)

    with open(args.idlist, encoding="utf-8") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    if args.limit:
        ids = ids[: args.limit]

    # resume: skip IDs already converted
    conv_marker = os.path.join(parq_dir, ".converted")
    converted = set()
    if os.path.exists(conv_marker):
        with open(conv_marker, encoding="utf-8") as f:
            converted = {ln.strip().lower() for ln in f if ln.strip()}
    pending = [i for i in ids if i.lower() not in converted]

    print(f"total={len(ids)} to_process={len(pending)} "
          f"(already converted={len(ids)-len(pending)}) jobs={args.jobs} chunk={args.chunk}")
    if not pending:
        print("nothing to do")
        return 0

    ok, skipped, failed, parse_failed, elapsed = asyncio.run(
        run(pending, raw_dir, parq_dir, args.chunk, args.jobs)
    )
    print(f"\ndone: ok={ok} skipped={skipped} failed={failed} parse_failed={parse_failed} "
          f"in {elapsed:.0f}s ({(ok+skipped+failed)/elapsed if elapsed else 0:.1f}/s)")
    if failed:
        print(f"failures logged in {os.path.join(parq_dir, '_failures.txt')}", file=sys.stderr)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
