# -*- coding: utf-8 -*-
"""
执行 analysis.sql
=================

准备数据（按 pipeline.clean() 的口径）→ 注册进 DuckDB → 逐条执行 analysis.sql。

为什么要有这个脚本：一份"只写不跑"的 SQL 在面试里很危险 ——
面试官会当场让你解释某个数字，你说不出来就露馅了。
这个脚本保证 analysis.sql 里的每个数字都是真跑出来的。

用法：
    "D:/venvs/lending-club/Scripts/python.exe" run_sql.py
    "D:/venvs/lending-club/Scripts/python.exe" run_sql.py --sample 200000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import duckdb
import pandas as pd

import pipeline as P

HERE = Path(__file__).parent
SQL_FILE = HERE / "analysis.sql"
CACHE = HERE / "data" / "_cache_raw_sample.pkl"     # 与 eda.py 共用同一份缓存
OUT_FILE = HERE / "sql_results.txt"
CLEAN_CACHE = HERE / "data" / "_cache_clean.pkl"


def build_loans(data_path: str | None, sample_n: int) -> pd.DataFrame:
    """拿到一份【和建模完全同口径】的干净数据。"""
    if CLEAN_CACHE.exists():
        df = pd.read_pickle(CLEAN_CACHE)
        print(f"[缓存] 命中 {CLEAN_CACHE.name}：{len(df):,} 行，坏账率 {df['bad_loan'].mean():.2%}")
        return df

    if CACHE.exists():
        raw = pd.read_pickle(CACHE)
        print(f"[缓存] 命中 {CACHE.name}：{len(raw):,} 行")
    else:
        path = P.resolve_data_path(data_path)
        keep = set(P.RAW_COLS)
        raw = pd.read_csv(path, usecols=lambda c: c in keep, low_memory=False)
        print(f"[加载] {path.name}：全量 {len(raw):,} 行")
        if len(raw) > sample_n:
            raw = raw.sample(n=sample_n, random_state=P.SEED).reset_index(drop=True)

    df = P.clean(raw)
    df.to_pickle(CLEAN_CACHE)
    print(f"[缓存] 已写入 {CLEAN_CACHE.name}")
    return df


def split_statements(sql_text: str) -> list[str]:
    """按分号切分语句。跳过只有注释/空白的片段。"""
    out = []
    for chunk in sql_text.split(";"):
        meaningful = [ln for ln in chunk.splitlines()
                      if ln.strip() and not ln.strip().startswith("--")]
        if meaningful:
            out.append(chunk.strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--sample", type=int, default=P.SAMPLE_N)
    ap.add_argument("--refresh", action="store_true", help="重建清洗缓存")
    args = ap.parse_args()

    if args.refresh and CLEAN_CACHE.exists():
        CLEAN_CACHE.unlink()

    print("\n" + "=" * 60)
    print("执行 analysis.sql（DuckDB）")
    print("=" * 60)

    df = build_loans(args.data, args.sample)

    con = duckdb.connect()          # 内存库，不落盘
    con.register("loans", df)
    n = con.execute("SELECT COUNT(*) FROM loans").fetchone()[0]
    print(f"[DuckDB] 已注册表 loans：{n:,} 行 × {len(df.columns)} 列\n")

    statements = split_statements(SQL_FILE.read_text(encoding="utf-8"))
    print(f"[执行] 共 {len(statements)} 条查询\n")

    lines: list[str] = []
    lines.append(f"analysis.sql 执行结果（{n:,} 行样本，坏账率 {df['bad_loan'].mean():.2%}）\n")
    ok = 0
    for i, stmt in enumerate(statements, 1):
        # 取该语句前面连续的注释作为标题
        title = ""
        for ln in stmt.splitlines():
            s = ln.strip()
            if s.startswith("-- Q"):
                title = s.lstrip("- ").strip()
                break
            if s and not s.startswith("--"):
                break

        header = f"{'=' * 78}\n[{i}/{len(statements)}] {title}\n{'=' * 78}"
        print(header)
        lines.append("\n" + header)
        try:
            t0 = time.time()
            res = con.execute(stmt).df()
            body = res.to_string(index=False, max_colwidth=28)
            print(body)
            print(f"-- {len(res)} 行，耗时 {time.time() - t0:.2f}s\n")
            lines.append(body)
            ok += 1
        except Exception as e:
            msg = f"[失败] {type(e).__name__}: {e}"
            print(msg + "\n")
            lines.append(msg)

    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print("=" * 60)
    print(f"完成：{ok}/{len(statements)} 条查询成功")
    print(f"结果已保存到 {OUT_FILE.name}")


if __name__ == "__main__":
    main()
