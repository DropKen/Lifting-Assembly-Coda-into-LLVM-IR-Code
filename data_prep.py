#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 JSON/NDJSON 中的 {"MI": "...", "MCInst": "..."} 样本清洗为训练用 JSONL：
  {"arch":"aarch64","mc_text":"...", "mi_text":"..."}

支持输入：
  1) JSON 数组：[ {...}, {...}, ... ]
  2) NDJSON：每行一个 JSON 对象（行尾可能有逗号）
  3) 纯文本里直接拼接了多个 {...}{...}（无逗号）——会尝试自动包裹成数组

还会处理：
  - 未转义控制字符（tab 等），统一为空格
  - 去除 MI 尾部 " :: ( ... )" 注释（可配置保留）
  - 去重
"""

import argparse, json, re, sys
from typing import List, Dict, Any

WS = re.compile(r"\s+")
COMMENT_TAIL = re.compile(r"\s*::\s*\(.*\)\s*$")  # 去掉 MI 尾注释

def sanitize_text(s: str) -> str:
    """把控制字符清掉/合并空白/规整标点空格。"""
    # 容忍转义：先反转义常见序列（如 \t \n）; 再消除实际控制字符
    s = s.encode('utf-8', 'backslashreplace').decode('unicode_escape')
    s = ''.join(ch if ord(ch) >= 32 or ch in '\t ' else ' ' for ch in s)
    s = s.replace('\t', ' ')
    s = WS.sub(' ', s)
    s = s.replace("[ ", "[").replace(" ]", "]").replace(" ,", ",")
    s = re.sub(r",\s*", ", ", s)
    s = s.replace("( ", "(").replace(" )", ")")
    return s.strip()

def normalize_mc(s: str, to_lower: bool=True) -> str:
    s = sanitize_text(s)
    return s.lower() if to_lower else s

def normalize_mi(s: str, drop_comment: bool=True) -> str:
    s = sanitize_text(s)
    if drop_comment:
        s = COMMENT_TAIL.sub("", s)
    return s

def parse_as_json_array(raw: str) -> List[Dict[str, Any]]:
    """优先尝试把整个文件当 JSON 加载（数组或对象）。"""
    try:
        obj = json.loads(raw, strict=False)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            # 若是 { "data": [ ... ] } 这种
            if "data" in obj and isinstance(obj["data"], list):
                return obj["data"]
            # 单对象也返回为单元素数组
            return [obj]
    except Exception:
        pass
    return []

def parse_as_ndjson(raw: str) -> List[Dict[str, Any]]:
    """行分割解析：每行一个 JSON，允许行尾逗号，strict=False。"""
    records = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # 去掉行尾多余逗号
        if line.endswith(","):
            line = line[:-1]
        try:
            obj = json.loads(line, strict=False)
        except Exception:
            continue
        records.append(obj)
    return records

def parse_as_concatenated_objects(raw: str) -> List[Dict[str, Any]]:
    """若是 {...}{...}{...} 直接拼接，尝试加逗号包成数组再解析。"""
    text = raw.strip()
    if not text:
        return []
    # 去掉文件结尾多余逗号，再把相邻对象用逗号隔开
    text = text.strip(",")
    wrapped = "[" + re.sub(r"}\s*{", "},{", text) + "]"
    try:
        obj = json.loads(wrapped, strict=False)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    return []

def load_records(path: str) -> List[Dict[str, Any]]:
    raw = open(path, "rb").read().decode("utf-8", errors="replace")
    # 1) 尝试 JSON/数组
    recs = parse_as_json_array(raw)
    if recs:
        return recs
    # 2) 尝试 NDJSON
    recs = parse_as_ndjson(raw)
    if recs:
        return recs
    # 3) 尝试拼接对象
    recs = parse_as_concatenated_objects(raw)
    if recs:
        return recs
    raise RuntimeError("无法从输入解析出任何 JSON 记录；请检查文件是否为合法 JSON/NDJSON。")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入 JSON 文件（数组/NDJSON/拼接均可）")
    ap.add_argument("--output", required=True, help="输出 JSONL（每行 arch/mc_text/mi_text）")
    ap.add_argument("--arch", default="aarch64")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--keep-mi-comment", action="store_true")
    ap.add_argument("--no-lower-mc", action="store_true")
    args = ap.parse_args()

    try:
        records = load_records(args.input)
    except Exception as e:
        print(f"[FATAL] 解析失败：{e}", file=sys.stderr)
        sys.exit(1)

    seen = set()
    kept = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for obj in records:
            if not isinstance(obj, dict):
                continue
            mi_raw = obj.get("MI", "")
            mc_raw = obj.get("MCInst", "")
            if not mi_raw or not mc_raw:
                continue
            mc = normalize_mc(mc_raw, to_lower=not args.no_lower_mc)
            mi = normalize_mi(mi_raw, drop_comment=not args.keep_mi_comment)
            if not mc or not mi:
                continue
            key = (mc, mi)
            if args.dedup and key in seen:
                continue
            seen.add(key)
            rec = {"arch": args.arch, "mc_text": mc, "mi_text": mi}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[DONE] 输入记录数≈{len(records)}，写出 {kept} 条到 {args.output}"
          + (f"（去重后剩 {len(seen)}）" if args.dedup else ""))

if __name__ == "__main__":
    main()
