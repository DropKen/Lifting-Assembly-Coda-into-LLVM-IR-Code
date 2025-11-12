# eval_batch.py —— 批量验证（无 datasets / 无 bitsandbytes），单卡，全精（bf16/fp16）
import argparse, json, re, os, math, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROMPT = "[arch] {arch}\n[context]\n{ctx}\n[query]\n{q}\n[task] 输出一条等价 MachineInstr\n"
WS = re.compile(r"\s+")

def nrm(s: str) -> str:
    s = s.replace("\t", " ").strip()
    s = WS.sub(" ", s)
    s = (s.replace("[ ", "[").replace(" ]", "]").replace(" ,", ","))
    return s.lower()

def clean_one_line(s: str) -> str:
    """
    只保留类似 LLVM MI 的一行，去掉 Markdown 围栏、行尾注释/路径、debug-location 等。
    """
    raw_lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not raw_lines:
        return ""
    def _strip_line(ln: str) -> str:
        ln = ln.strip().strip("`，。；;！!？? ")
        ln = re.split(r"(?:;|//|#)", ln, 1)[0].strip()  # 行尾注释
        ln = re.sub(r"\bdebug-location\s*!?\d+\b.*$", "", ln, flags=re.IGNORECASE)
        ln = re.sub(r"\s+", " ", ln).strip()
        return ln
    mi_pat = re.compile(
        r"\b(MOVZ?K?|MOVN|MOV|ADD|ADDX|SUB|SUBX|LDRW?|STRW?|RET|ADR|ADRP|CBNZ?|CBZ|B\w*)\b",
        re.IGNORECASE,
    )
    for ln in raw_lines:
        ln2 = _strip_line(ln)
        if mi_pat.search(ln2):
            return ln2
    return _strip_line(raw_lines[0])

def relower_mi_to_mc(mi: str) -> str:
    """
    把常见 MI 形式兜成 MCInst 文本（尽量覆盖你任务里高频模式）。
    覆盖不到就返回空串，让评测记为错误样本，后续放到 hard pool 里复盘。
    """
    s = mi.strip()

    # movzwi -> mov wN, #imm
    m = re.match(r".*?(\$?w\d+)\s*=\s*movzwi\s+(\d+)\s*,\s*0\b", s, re.I)
    if m:
        w, imm = m.groups()
        return f"mov {w.replace('$','').lower()}, #{imm}"

    # addwrs -> add wdst, wa, wb  （shift=0）
    m = re.match(r"\s*\$?(w\d+)\s*=\s*addwrs\s+.*\$(w\d+).*\$(w\d+).*,\s*0\b", s, re.I)
    if m:
        d, a, b = m.groups()
        return f"add {d.lower()}, {a.lower()}, {b.lower()}"

    # add wzr + imm == mov
    m = re.match(r"\s*add\s+(w\d+)\s*,\s*wzr\s*,\s*#?(\d+)\s*$", s, re.I)
    if m:
        w, imm = m.groups()
        return f"mov {w.lower()}, #{imm}"

    # STRWui/LDRWui （word-offset -> byte offset *4）
    m = re.search(r"\bstrwui\b.*\$(w\d+)\s*,\s*\$sp\s*,\s*(\d+)", s, re.I)
    if m:
        w, off = m.groups()
        return f"str {w.lower()}, [sp, #{int(off)*4}]"
    m = re.match(r"\s*(?:renamable\s+)?(\$w\d+)\s*=\s*ldrwui\s+\$sp\s*,\s*(\d+)", s, re.I)
    if m:
        w, off = m.groups()
        return f"ldr {w.lower().replace('$','')}, [sp, #{int(off)*4}]"

    # 帧设置/销毁 & ret
    if re.search(r"\bsubxri\s+\$sp\s*,\s*16\b", s, re.I): return "sub sp, sp, #16"
    if re.search(r"\baddxri\s+\$sp\s*,\s*16\b", s, re.I): return "add sp, sp, #16"
    if re.match(r"^\s*ret\b", s, re.I):                  return "ret"

    # 兜一些常见“更接近 MCInst”的 MI 输出（不含 $）
    m = re.match(r"^\s*mov\s+(w\d+)\s*,\s*#?(\d+)\s*$", s, re.I)
    if m:
        w, imm = m.groups()
        return f"mov {w.lower()}, #{imm}"

    m = re.match(r"^\s*add\s+(w\d+)\s*,\s*(w\d+)\s*,\s*(w\d+)\s*$", s, re.I)
    if m:
        d,a,b = m.groups()
        return f"add {d.lower()}, {a.lower()}, {b.lower()}"

    return ""

def read_json_or_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1); f.seek(0)
        if first == "[":
            data = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data

def load_model(base_model: str, adapter: str, cuda: str, bf16: bool, fp16: bool):
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    use_bf16 = bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = (not use_bf16) and fp16
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    base.eval()

    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return tok, model, device, dtype

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="训练时用的同款基座")
    ap.add_argument("--adapter", required=True, help="LoRA 目录")
    ap.add_argument("--data", required=True, help="JSONL/JSON，含 arch/mc_text/mi_text")
    ap.add_argument("--save_hard", default="hard_examples.jsonl")
    ap.add_argument("--save_pred", default="predictions.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--cuda", default="0")
    args = ap.parse_args()

    tok, model, device, dtype = load_model(args.base_model, args.adapter, args.cuda, args.bf16, args.fp16)
    data = read_json_or_jsonl(args.data)
    if args.limit > 0:
        data = data[:args.limit]

    total, ok = 0, 0
    hard = []
    out_f = open(args.save_pred, "w", encoding="utf-8")

    for rec in data:
        arch = rec.get("arch", "aarch64")
        mc_gold = rec["mc_text"]
        prompt = PROMPT.format(arch=arch, ctx="", q=mc_gold)

        # 与训练一致：不加 special tokens
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
        gen_ids = out[0, input_len:]
        text = tok.decode(gen_ids, skip_special_tokens=True).strip()
        # 清理潜在回显（保险）
        text = re.split(r"(?:\r?\n)?(?:Human|User|Assistant|系统提示)\s*[:：]", text, maxsplit=1, flags=re.I)[0].strip()

        pred_mi = clean_one_line(text)
        mc_hat = relower_mi_to_mc(pred_mi)

        good = (nrm(mc_hat) == nrm(mc_gold))
        total += 1
        ok += int(good)

        out_f.write(json.dumps({
            "arch": arch,
            "mc_gold": mc_gold,
            "mi_gold": rec.get("mi_text", ""),
            "mi_pred": pred_mi,
            "mc_pred": mc_hat,
            "match": bool(good)
        }, ensure_ascii=False) + "\n")

        if not good:
            hard.append({
                "arch": arch,
                "mc_text": mc_gold,
                "mi_gold": rec.get("mi_text", ""),
                "mi_pred": pred_mi,
                "mc_pred": mc_hat
            })

    out_f.close()
    acc = ok / max(total, 1)
    print(f"Roundtrip accuracy: {ok}/{total} = {acc:.4f}")

    if hard:
        with open(args.save_hard, "w", encoding="utf-8") as f:
            for h in hard:
                f.write(json.dumps(h, ensure_ascii=False) + "\n")
        print(f"Saved {len(hard)} hard examples to {args.save_hard}")

if __name__ == "__main__":
    main()
