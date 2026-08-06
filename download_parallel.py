#!/usr/bin/env python3
"""Asynchronous parallel download of PDB mmCIF files using aiohttp.

Features:
  * Asynchronous I/O with aiohttp (higher concurrency than threading)
  * Configurable concurrency via --jobs (semaphore limits active downloads)
  * Resume (skips files already present and non-trivial)
  * Retries with exponential backoff (3 attempts)
  * Failure log (data/raw/_failures.txt)
  * Progress bar with tqdm (real-time rate, success/fail counts)

Usage:
    python3 download_parallel.py pdb_ids.txt --out data/raw --jobs 64 --limit 100
"""

import argparse
import asyncio
import aiohttp
import aiofiles
import os
import sys
import time
from tqdm import tqdm

BASE_URL = "https://files.wwpdb.org/download"   # 使用官方推荐域名
SUFFIX = ".cif.gz"
RETRIES = 3
MIN_VALID_SIZE = 100
TIMEOUT = aiohttp.ClientTimeout(total=90)

# 全局并发信号量，控制同时进行的下载数量
semaphore = None
fail_log_path = None


async def fetch_one(session: aiohttp.ClientSession, ident: str, out_dir: str) -> tuple[str, bool, str]:
    """Download one mmCIF file asynchronously, with skip and retry logic."""
    dest = os.path.join(out_dir, ident.lower() + SUFFIX)
    # 检查是否已存在且文件大小正常
    if os.path.exists(dest) and os.path.getsize(dest) > MIN_VALID_SIZE:
        return ident, True, "exists"

    url = f"{BASE_URL}/{ident.lower()}{SUFFIX}"
    for attempt in range(RETRIES):
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    # 如果返回非200，尝试读取错误信息（但不要阻塞太久）
                    text = await resp.text(errors='ignore')
                    if "xml" in text[:20].lower():
                        return ident, False, "error-response"
                    return ident, False, f"HTTP {resp.status}"
                data = await resp.read()
                if len(data) < MIN_VALID_SIZE:
                    return ident, False, "too-small"
                # 写入临时文件
                tmp = dest + ".part"
                async with aiofiles.open(tmp, 'wb') as f:
                    await f.write(data)
                os.replace(tmp, dest)
                return ident, True, "ok"
        except Exception as e:
            if attempt < RETRIES - 1:
                wait = 2 ** attempt  # 指数退避：1, 2, 4 秒
                await asyncio.sleep(wait)
            else:
                return ident, False, f"{type(e).__name__}: {e}"
    return ident, False, "unknown"


async def worker(session: aiohttp.ClientSession, ident: str, out_dir: str, pbar: tqdm | None):
    """Wrapper for fetch_one with semaphore control and progress update."""
    global semaphore
    async with semaphore:
        ident, success, why = await fetch_one(session, ident, out_dir)
        # 更新统计（通过返回值或外部计数器实现，为了简单，我们直接在外部处理）
        return ident, success, why


async def main_async(ids: list[str], out_dir: str, jobs: int) -> tuple[int, int, int]:
    global semaphore, fail_log_path
    semaphore = asyncio.Semaphore(jobs)
    fail_log_path = os.path.join(out_dir, "_failures.txt")

    os.makedirs(out_dir, exist_ok=True)
    total = len(ids)
    ok = skipped = failed = 0
    t0 = time.time()

    # 使用 tqdm 异步更新（需要手动维护）
    pbar = tqdm(total=total, desc="Downloading", unit="file")

    # 创建 aiohttp session（可复用连接池）
    connector = aiohttp.TCPConnector(limit=jobs * 2, limit_per_host=jobs)
    async with aiohttp.ClientSession(connector=connector, timeout=TIMEOUT) as session:
        # 创建所有任务
        tasks = [asyncio.create_task(worker(session, ident, out_dir, pbar)) for ident in ids]
        # 使用 asyncio.as_completed 逐个获取结果并更新进度
        for coro in asyncio.as_completed(tasks):
            ident, success, why = await coro
            if success and why == "exists":
                skipped += 1
            elif success:
                ok += 1
            else:
                failed += 1
                # 写入失败日志（异步写文件可能影响性能，这里用普通同步写，因为很少发生）
                with open(fail_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{ident}\t{why}\n")
            # 更新进度条
            elapsed = time.time() - t0
            rate = (ok + skipped + failed) / elapsed if elapsed > 0 else 0
            pbar.set_postfix(ok=ok, skip=skipped, fail=failed, rate=f"{rate:.1f}/s", refresh=False)
            pbar.update(1)

    pbar.close()
    elapsed = time.time() - t0
    return ok, skipped, failed, elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("idlist")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--jobs", type=int, default=32, help="concurrent downloads (semaphore limit)")
    ap.add_argument("--limit", type=int, default=0, help="download only first N IDs")
    args = ap.parse_args()

    with open(args.idlist, encoding="utf-8") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    if args.limit:
        ids = ids[: args.limit]

    print(f"Total: {len(ids)} structures, output: {args.out}, concurrency: {args.jobs}")
    ok, skipped, failed, elapsed = asyncio.run(main_async(ids, args.out, args.jobs))
    print(f"Done: ok={ok}, skipped={skipped}, failed={failed} in {elapsed:.0f}s, rate={(ok+skipped+failed)/elapsed:.1f}/s")
    if failed:
        print(f"Failures logged in {os.path.join(args.out, '_failures.txt')}", file=sys.stderr)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())