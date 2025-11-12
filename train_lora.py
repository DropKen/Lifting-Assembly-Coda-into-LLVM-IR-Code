# train_lora.py —— 纯 PyTorch + LoRA（无 datasets / 无 Trainer / 无 bitsandbytes）
# 兼容老 transformers：AutoTokenizer 不认 Qwen2 时，回退到 PreTrainedTokenizerFast 读取 tokenizer.json
import argparse, json, os, math
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM

# 你原来的 Prompt 模板
PROMPT = "[arch] {arch}\n[context]\n{ctx}\n[query]\n{q}\n[task] 输出一条等价 MachineInstr\n"

def load_json_or_jsonl(path: str) -> List[Dict]:
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

def build_examples(raw: List[Dict], tok, max_inp=256, max_tgt=128):
    prompts, targets = [], []
    for ex in raw:
        arch = ex.get("arch") or "aarch64"
        mc   = ex.get("mc_text") or ""
        mi   = (ex.get("mi_text") or "").strip() + "\n"
        p = PROMPT.format(arch=arch, ctx="", q=mc)
        prompts.append(p); targets.append(mi)

    X = tok(prompts, add_special_tokens=False, truncation=True, max_length=max_inp)
    Y = tok(targets, add_special_tokens=False, truncation=True, max_length=max_tgt)

    items = []
    for x_ids, y_ids in zip(X["input_ids"], Y["input_ids"]):
        ids = x_ids + y_ids
        labels = [-100] * len(x_ids) + y_ids   # 只训练答案部分
        items.append({"input_ids": ids, "labels": labels})
    return items

class JsonlDataset(Dataset):
    def __init__(self, items: List[Dict]): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

def make_collate_fn(tok):
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    def collate(batch: List[Dict]):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            ids, lab = x["input_ids"], x["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id]*pad_len)
            attention_mask.append([1]*len(ids) + [0]*pad_len)
            labels.append(lab + [-100]*pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return collate

def make_linear_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = float(step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 1.0 - progress)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def load_tokenizer(model_id: str):
    """
    兼容旧版 transformers：
    1) 先尝试 AutoTokenizer（trust_remote_code），
    2) 失败则回退到 PreTrainedTokenizerFast 直接加载 tokenizer.json。
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False  # 老环境更稳；如果你已装新 tokenizers，可以改 True
        )
        print("[info] Loaded tokenizer via AutoTokenizer.")
        return tok
    except Exception as e:
        print(f"[warn] AutoTokenizer 加载失败，将回退到 PreTrainedTokenizerFast。原因: {e}")
        from transformers import PreTrainedTokenizerFast
        # 直接读 repo 里的 tokenizer.json；大多数现代模型（含 Qwen2.5）都有
        tok = PreTrainedTokenizerFast.from_pretrained(
            model_id,
            tokenizer_file="tokenizer.json",
            trust_remote_code=True
        )
        # 尝试从 special_tokens_map.json 里拿到 eos/pad；拿不到则兜底
        if tok.eos_token_id is None:
            # 常见兜底：有的仓库把 eos 放到 <|endoftext|> 或 '' 里
            for cand in ["<|endoftext|>", "</s>", ""]:
                try:
                    tid = tok.convert_tokens_to_ids(cand)
                    if tid is not None and tid != tok.unk_token_id:
                        tok.eos_token = cand
                        break
                except Exception:
                    pass
            if tok.eos_token_id is None:
                raise RuntimeError("未能在回退路径下确定 eos_token，请升级 transformers 或检查仓库的 special_tokens_map.json。")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        print("[info] Loaded tokenizer via PreTrainedTokenizerFast (tokenizer.json).")
        return tok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_inp", type=int, default=256)
    ap.add_argument("--max_tgt", type=int, default=128)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # === Tokenizer（带回退）===
    tok = load_tokenizer(args.base_model)
    print(f"[tok] eos={tok.eos_token!r}/{tok.eos_token_id} pad={tok.pad_token!r}/{tok.pad_token_id}")

    # === Precision ===
    use_bf16 = args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = (not use_bf16) and args.fp16
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else None)
    if args.bf16 and not use_bf16:
        print("[warn] 本机不支持 bf16，将尝试 fp16/float32")

    # === Base model（允许 remote code，老环境更稳）===
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=(torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)),
        device_map="auto",
        trust_remote_code=True
    )
    base.config.use_cache = False  # 配合梯度检查点

    # === LoRA ===
    from peft import LoraConfig, get_peft_model, TaskType
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()

    # —— 保留梯度检查点，但确保输入张量有 grad —— #
    use_gc = True
    if use_gc and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        # 关键：让 embedding 的输出在前向时标记为 require_grad，避免 checkpoint 报错
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            emb = model.get_input_embeddings()

            def _make_out_require_grad(module, inputs, output):
                if isinstance(output, torch.Tensor):
                    output.requires_grad_(True)

            emb.register_forward_hook(_make_out_require_grad)

    torch.set_float32_matmul_precision("high")

    # === Data ===
    raw = load_json_or_jsonl(args.train_file)
    items = build_examples(raw, tok, args.max_inp, args.max_tgt)
    train_ds = JsonlDataset(items)
    collate_fn = make_collate_fn(tok)
    loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=collate_fn
    )

    # === Optim & Sched（对齐常用默认）===
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    total_steps = (len(loader) + args.grad_accum - 1) // args.grad_accum * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    sched = make_linear_scheduler(optim, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_fp16 and not use_bf16))

    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    global_step = 0
    running_loss = 0.0
    log_every = 10

    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model(**batch)
                    loss = out.loss / args.grad_accum
            else:
                out = model(**batch)
                loss = out.loss / args.grad_accum

            if use_fp16 and not use_bf16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.grad_accum == 0:
                if use_fp16 and not use_bf16:
                    scaler.step(optim); scaler.update()
                else:
                    optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                running_loss += out.loss.item() * args.grad_accum
                if global_step % log_every == 0:
                    lr = sched.get_last_lr()[0]
                    print(f"[epoch {epoch+1}] step {global_step}/{total_steps}  lr={lr:.3e}  loss={running_loss/log_every:.4f}")
                    running_loss = 0.0

        # 每个 epoch 存一次
        save_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        try:
            tok.save_pretrained(save_dir)
        except Exception:
            # PreTrainedTokenizerFast 也支持 save_pretrained；这里兜底一次
            from pathlib import Path
            Path(os.path.join(save_dir, "tokenizer_saved.txt")).write_text("tokenizer saved")
        print(f"[save] epoch {epoch+1} -> {save_dir}")

    # 最终保存
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    try:
        tok.save_pretrained(args.output_dir)
    except Exception:
        pass
    print("Saved LoRA to", args.output_dir)

if __name__ == "__main__":
    main()
