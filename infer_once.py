import argparse, os, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROMPT = "[arch] {arch}\n[context]\n{ctx}\n[query]\n{q}\n[task] 输出一条等价 MachineInstr\n"

def clean_one_line(s: str) -> str:
    """
    只保留类似 LLVM MI 的一行，去掉 Markdown 围栏、行尾注释/路径、debug-location 等元信息。
    """
    # 取非空行
    raw_lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not raw_lines:
        return ""

    def _strip_line(ln: str) -> str:
        # 去掉围栏和左/右侧多余标点
        ln = ln.strip().strip("`，。；;！!？? ")
        # 砍掉常见注释：; // #
        ln = re.split(r"(?:;|//|#)", ln, 1)[0].strip()
        # 去掉 debug-location !1234 之后的任何内容（保险起见）
        ln = re.sub(r"\bdebug-location\s*!?\d+\b.*$", "", ln, flags=re.IGNORECASE)
        # 压缩多空格
        ln = re.sub(r"\s+", " ", ln).strip()
        return ln

    # 先找最像 MI 的行
    mi_pat = re.compile(
        r"\b(MOVZ?K?|MOVN|ADD|ADDX|SUB|SUBX|LDRW?|STRW?|RET|ADR|ADRP|CBNZ?|B\w*)\b",
        re.IGNORECASE,
    )
    for ln in raw_lines:
        ln2 = _strip_line(ln)
        if mi_pat.search(ln2):
            return ln2

    # 兜底：拿第一行做最小清洗
    return _strip_line(raw_lines[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="训练好的 LoRA 目录")
    ap.add_argument("--base_model", required=True, help="必须与训练时的基座一致，例如 Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--query", required=True)
    ap.add_argument("--context", default="")
    ap.add_argument("--arch", default="aarch64")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--cuda", default="0", help="用哪张卡，默认 0")
    args = ap.parse_args()

    # —— 设备与精度 —— #
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    use_bf16 = args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = (not use_bf16) and args.fp16
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    # —— 分词器：与训练一致，信任远程代码；必要时可改 use_fast=False —— #
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # —— 加载基座（不使用 device_map / bnb；固定到单卡） —— #
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    base.eval()

    # —— 挂载 LoRA 适配器（必须同一基座族/形状！） —— #
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    # —— 构造与训练一致的提示词 —— #
    prompt = PROMPT.format(arch=args.arch, ctx=args.context, q=args.query)
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
    text = tok.decode(gen_ids, skip_special_tokens=True)
    mi = clean_one_line(text)

    print(mi)

if __name__ == "__main__":
    main()
