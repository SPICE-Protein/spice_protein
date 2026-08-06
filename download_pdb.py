#!/usr/bin/env python3
"""One-shot async download + convert pipeline for SPICE PDB data.

Supports separate download-only and convert-only modes with resumable progress.

Usage (Python 3.10+, run on Colab / server):
    pip install aiohttp aiofiles gemmi polars pyarrow tqdm

    # Download only
    python download_pdb.py pdb_ids.txt --out data --jobs 64 --download-only

    # Convert only (all raw files)
    python download_pdb.py --out data --convert-only

    # Convert only for specific IDs
    python download_pdb.py pdb_ids.txt --out data --convert-only

    # Full pipeline (download then convert)
    python download_pdb.py pdb_ids.txt --out data --jobs 64 --chunk 5000

Outputs:
    data/raw/...                 (transient, deleted after conversion unless --keep-raw)
    data/parquet/entries_*.parquet   (one row per structure)
    data/parquet/atoms_*.parquet     (one row per atom, long format)
    data/parquet/.converted      (resume marker: already-processed IDs)
    data/parquet/state.pkl       (pickle checkpoint: converted set)
    data/parquet/_failures.txt   (download/parse failures)

Resume:
    - Download: raw files already present are skipped.
    - Convert: IDs in .converted are skipped; interrupted conversion restarts safely.
"""

import argparse
import asyncio
import gzip
import os
import pickle
import re
import signal
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

BASE_URL = "https://files.wwpdb.org/download"
SUFFIX = ".cif.gz"
RETRIES = 3
MIN_VALID_SIZE = 100
TIMEOUT = aiohttp.ClientTimeout(total=90)

PH_MIN, PH_MAX = 0.0, 14.0
TEMP_MIN, TEMP_MAX = 150.0, 400.0

# ---------------------------------------------------------------------------
# 1-letter amino-acid map
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
# Ionic-strength heuristic
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
# Parquet schemas
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


def _atomic_write_parquet(df, path):
    tmp = path + ".tmp"
    df.write_parquet(tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Checkpoint for converted IDs
# ---------------------------------------------------------------------------
STATE_FILE = "state.pkl"
SAVE_EVERY = 200


class Checkpoint:
    def __init__(self, path):
        self.path = path
        self.converted = set()

    @classmethod
    def open(cls, parq_dir):
        cp = cls(os.path.join(parq_dir, STATE_FILE))
        cp.load()
        return cp

    def load(self):
        try:
            with open(self.path, "rb") as f:
                d = pickle.load(f)
            self.converted = {str(x).lower() for x in d.get("converted", ())}
        except (OSError, EOFError, ValueError, pickle.UnpicklingError):
            pass

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"converted": sorted(self.converted)}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.path)


def write_sliding_chunk(entries, atoms, ids, parq_dir, shard_no, keep_raw, raw_dir, conv_marker, state):
    """Write one chunk to Parquet atomically, update .converted and state."""
    shard_no += 1
    atoms_path = os.path.join(parq_dir, f"atoms_shard_{shard_no:04d}.parquet")
    entries_path = os.path.join(parq_dir, f"entries_shard_{shard_no:04d}.parquet")
    _atomic_write_parquet(pl.DataFrame(atoms, schema=ATOM_SCHEMA), atoms_path)
    _atomic_write_parquet(pl.DataFrame(entries, schema=ENTRY_SCHEMA), entries_path)

    # Update converted marker
    with open(conv_marker, "a", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")
    state.converted.update(i.lower() for i in ids)
    state.save()

    # Delete raw files unless keep_raw
    if not keep_raw:
        for ident in ids:
            src = os.path.join(raw_dir, ident.lower() + SUFFIX)
            if os.path.exists(src):
                os.remove(src)
    return shard_no


# ---------------------------------------------------------------------------
# Download-only
# ---------------------------------------------------------------------------
async def download_ids(ids, raw_dir, jobs, stop_event=None):
    """Download only, skip if raw file already exists."""
    sem = asyncio.Semaphore(jobs)
    connector = aiohttp.TCPConnector(limit=jobs * 2, limit_per_host=jobs)
    fail_log = os.path.join(raw_dir, "_download_failures.txt")
    ok = skipped = failed = 0

    pbar = tqdm(total=len(ids), unit="entry", disable=tqdm is None)
    t0 = time.time()

    async with aiohttp.ClientSession(connector=connector, timeout=TIMEOUT) as session:
        # Use a queue to process IDs with semaphore
        tasks = []
        for ident in ids:
            if stop_event and stop_event.is_set():
                break
            # Wait for semaphore before creating task
            async with sem:
                task = asyncio.create_task(download_one(session, ident, raw_dir))
                tasks.append(task)

        for task in asyncio.as_completed(tasks):
            if stop_event and stop_event.is_set():
                break
            try:
                status, dest = await task
            except Exception as e:
                status = f"exception: {e}"
                failed += 1
                pbar.update(1)
                continue

            key = status.split(":")[0]
            if key in ("ok", "exists"):
                if key == "exists":
                    skipped += 1
                else:
                    ok += 1
            else:
                failed += 1
                # Log failure
                ident = os.path.basename(dest).replace(SUFFIX, "")
                with open(fail_log, "a", encoding="utf-8") as f:
                    f.write(f"{ident}\t{status}\n")
            pbar.update(1)

    elapsed = time.time() - t0
    if pbar:
        pbar.close()
    return ok, skipped, failed, elapsed


async def download_one(session, ident, raw_dir):
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
                await asyncio.sleep(2 ** attempt)
            else:
                return f"{type(e).__name__}: {e}", dest
    return "unknown", dest


# ---------------------------------------------------------------------------
# Convert-only
# ---------------------------------------------------------------------------
async def convert_all(raw_dir, parq_dir, chunk, ids=None, keep_raw=False,
                      flush_every=0.0, stop_event=None, state=None):
    """Convert all raw files (or subset) to Parquet, skip already converted."""
    if state is None:
        state = Checkpoint.open(parq_dir)

    # Load converted from .converted file (for backward compatibility)
    conv_marker = os.path.join(parq_dir, ".converted")
    if os.path.exists(conv_marker):
        with open(conv_marker, encoding="utf-8") as f:
            state.converted |= {ln.strip().lower() for ln in f if ln.strip()}

    # Gather raw files
    if ids is None:
        # Scan all .cif.gz files
        raw_files = [f for f in os.listdir(raw_dir) if f.endswith(SUFFIX)]
        ids = [f[:-len(SUFFIX)].lower() for f in raw_files]
    else:
        ids = [i.lower() for i in ids]
        # Only include files that exist
        existing = []
        for i in ids:
            path = os.path.join(raw_dir, i + SUFFIX)
            if os.path.exists(path) and os.path.getsize(path) > MIN_VALID_SIZE:
                existing.append(i)
        ids = existing

    # Remove already converted
    pending = [i for i in ids if i not in state.converted]
    if not pending:
        print("No pending files to convert.")
        return 0, 0, 0, 0

    # Determine starting shard number
    existing_shards = [f for f in os.listdir(parq_dir) if f.startswith("entries_shard_")]
    shard_no = len(existing_shards)

    pbar = tqdm(total=len(pending), unit="entry", disable=tqdm is None)
    ok = parse_failed = failed = 0
    buf_entries, buf_atoms, buf_ids = [], [], []
    last_flush = time.time()
    t0 = last_flush

    def flush():
        nonlocal shard_no, last_flush, ok
        if buf_entries:
            shard_no = write_sliding_chunk(
                buf_entries, buf_atoms, buf_ids, parq_dir, shard_no,
                keep_raw, raw_dir, conv_marker, state
            )
            ok += len(buf_ids)
            buf_entries.clear()
            buf_atoms.clear()
            buf_ids.clear()
            state.save()
        last_flush = time.time()

    for ident in pending:
        if stop_event and stop_event.is_set():
            break
        path = os.path.join(raw_dir, ident + SUFFIX)
        entry, atoms = await asyncio.to_thread(parse_block, path)
        if entry is not None:
            buf_entries.append(entry)
            buf_atoms.extend(atoms)
            buf_ids.append(ident)
            # Flush if full or timed out
            if len(buf_entries) >= chunk:
                flush()
            elif flush_every > 0 and (time.time() - last_flush) >= flush_every:
                flush()
        else:
            # Parse failed: move to failure dir
            parse_failed += 1
            fail_dir = os.path.join(raw_dir, "_parse_failed")
            os.makedirs(fail_dir, exist_ok=True)
            src = path
            if os.path.exists(src):
                os.replace(src, os.path.join(fail_dir, ident + SUFFIX))
        pbar.update(1)

    # Final flush
    if buf_entries:
        flush()

    elapsed = time.time() - t0
    if pbar:
        pbar.close()
    return ok, parse_failed, failed, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('idlist', nargs='?', help='file with PDB IDs (required for download/all mode)')
    ap.add_argument('--out', default='data', help='output directory (default: data)')
    ap.add_argument('--jobs', type=int, default=32, help='concurrent downloads')
    ap.add_argument('--chunk', type=int, default=5000, help='structures per parquet shard')
    ap.add_argument('--limit', type=int, default=0, help='only first N IDs (download only)')
    ap.add_argument('--flush-every', type=float, default=0.0,
                    help='seconds between forced flushes during conversion')
    ap.add_argument('--download-only', action='store_true', help='only download, do not convert')
    ap.add_argument('--convert-only', action='store_true', help='only convert, do not download')
    ap.add_argument('--keep-raw', action='store_true', help='keep raw files after conversion')
    args = ap.parse_args()

    # Mutual exclusivity
    if args.download_only and args.convert_only:
        print("Error: --download-only and --convert-only are mutually exclusive.", file=sys.stderr)
        return 1

    raw_dir = os.path.join(args.out, 'raw')
    parq_dir = os.path.join(args.out, 'parquet')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(parq_dir, exist_ok=True)

    # Load IDs if provided
    ids = None
    if args.idlist:
        with open(args.idlist, encoding='utf-8') as f:
            ids = [ln.strip() for ln in f if ln.strip()]
        if args.limit:
            ids = ids[:args.limit]

    # Download-only
    if args.download_only:
        if not ids:
            print("Error: --download-only requires an ID list file.", file=sys.stderr)
            return 1
        print(f"Download mode: {len(ids)} IDs, jobs={args.jobs}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        stop_event = asyncio.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except (NotImplementedError, RuntimeError):
                pass
        try:
            ok, skipped, failed, elapsed = loop.run_until_complete(
                download_ids(ids, raw_dir, args.jobs, stop_event)
            )
        finally:
            loop.close()
        print(f"Download done: ok={ok}, skipped={skipped}, failed={failed} in {elapsed:.0f}s")
        return 0 if failed == 0 else 1

    # Convert-only
    if args.convert_only:
        print(f"Convert mode: scanning {raw_dir}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        stop_event = asyncio.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except (NotImplementedError, RuntimeError):
                pass
        try:
            ok, parse_failed, failed, elapsed = loop.run_until_complete(
                convert_all(raw_dir, parq_dir, args.chunk, ids=ids,
                            keep_raw=args.keep_raw, flush_every=args.flush_every,
                            stop_event=stop_event)
            )
        finally:
            loop.close()
        print(f"Convert done: converted={ok}, parse_failed={parse_failed}, failed={failed} in {elapsed:.0f}s")
        return 0 if (parse_failed + failed) == 0 else 1

    # Full pipeline: download then convert
    if not ids:
        print("Error: full pipeline requires an ID list file.", file=sys.stderr)
        return 1

    print(f"Full pipeline: {len(ids)} IDs, jobs={args.jobs}, chunk={args.chunk}")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop_event = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):
            pass

    try:
        # Download phase
        print("\n=== Download phase ===")
        ok_dl, skipped_dl, failed_dl, elapsed_dl = loop.run_until_complete(
            download_ids(ids, raw_dir, args.jobs, stop_event)
        )
        print(f"Download done: ok={ok_dl}, skipped={skipped_dl}, failed={failed_dl} in {elapsed_dl:.0f}s")
        if failed_dl > 0:
            print(f"Warning: {failed_dl} download failures. Conversion will skip missing files.")

        # Convert phase
        print("\n=== Convert phase ===")
        ok_cv, parse_failed, failed_cv, elapsed_cv = loop.run_until_complete(
            convert_all(raw_dir, parq_dir, args.chunk, ids=ids,
                        keep_raw=args.keep_raw, flush_every=args.flush_every,
                        stop_event=stop_event)
        )
        print(f"Convert done: converted={ok_cv}, parse_failed={parse_failed}, failed={failed_cv} in {elapsed_cv:.0f}s")

    finally:
        loop.close()

    total_fail = failed_dl + parse_failed + failed_cv
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())