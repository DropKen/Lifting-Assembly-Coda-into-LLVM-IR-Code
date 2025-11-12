import os, json, random, subprocess, sys, shlex, platform, time

BASE = "models/Qwen2.5-Coder-3B-Instruct"
DATA = "new_data.jsonl"
EXP  = "experiments/Qwen"
OFF  = "offload"
TARGET = 0.95
REPLAY_N = 5000
MAX_ROUNDS = 5
LR = [2e-4, 1.5e-4, 1e-4, 8e-5, 6e-5]

os.makedirs(EXP, exist_ok=True)
os.makedirs(OFF, exist_ok=True)

IS_WINDOWS = platform.system().lower().startswith("win")

def run(cmd, log_path=None):
    """流式打印 + 失败时完整输出。Windows 下用 shell=True 防止 shlex 兼容问题。"""
    print(">>", cmd)
    sys.stdout.flush()
    if log_path:
        f = open(log_path, "w", encoding="utf-8")
    else:
        f = None
    try:
        if IS_WINDOWS:
            # Windows 用 shell=True 更稳（路径里有 \ 与空格时）
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        else:
            p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out_lines = []
        for line in p.stdout:
            line = line.rstrip("\n")
            out_lines.append(line)
            print(line)
            if f: f.write(line + "\n")
        p.wait()
        if p.returncode != 0:
            print(f"[ERROR] Command failed with code {p.returncode}")
            if f: f.close()
            raise subprocess.CalledProcessError(p.returncode, cmd, "\n".join(out_lines))
        if f: f.close()
        return "\n".join(out_lines)
    except Exception as e:
        if f: f.close()
        raise

def sample_replay(src, n, dst):
    with open(src, "r", encoding="utf-8") as f:
        lines = f.readlines()
    pick = lines if n >= len(lines) else random.sample(lines, n)
    with open(dst, "w", encoding="utf-8") as w:
        w.writelines(pick)

def merge_and_dedup(replay, hard, dst):
    seen=set()
    def emit(o, w):
        k = o.get("mc_text","")+"<<<>>>"+o.get("mi_text","")
        if k in seen: return
        seen.add(k); w.write(json.dumps(o, ensure_ascii=False)+"\n")
    with open(dst,"w",encoding="utf-8") as w:
        for path in (replay, hard):
            if not os.path.exists(path): continue
            for line in open(path,"r",encoding="utf-8"):
                line=line.strip()
                if not line: continue
                try:
                    o=json.loads(line)
                    if "mi_text" not in o and "mi_gold" in o:
                        o={"arch":o.get("arch","aarch64"),"mc_text":o["mc_text"],"mi_text":o["mi_gold"]}
                    emit(o,w)
                except: pass

# 先做基本存在性检查，避免无谓失败
for p in [BASE, DATA]:
    if not os.path.exists(p):
        print(f"[FATAL] 路径不存在：{p}")
        sys.exit(1)

mix = DATA
for r in range(MAX_ROUNDS):
    outdir = f"{EXP}/ckpt-r{r}"
    hard   = f"{EXP}/hard-r{r}.jsonl"
    lr     = LR[r] if r < len(LR) else LR[-1]
    os.makedirs(outdir, exist_ok=True)

    # 训练
    train_cmd = (
        f"python train_lora.py "
        f"--train_file {mix} "
        f"--output_dir {outdir} "
        f"--base_model {BASE} "
        f"--epochs 1 --batch 1 --grad_accum 16 --lr {lr}"
    )
    run(train_cmd, log_path=f"{outdir}/train.log")

    # 评测
    eval_cmd = (
        f"python eval_roundtrip.py "
        f"--base_model {BASE} "
        f"--adapter {outdir} "
        f"--data {DATA} "
        f"--save_hard {hard} "
        f"--offload_dir {OFF}"
    )
    eval_out = run(eval_cmd, log_path=f"{outdir}/eval.log")

    # 解析 acc
    acc = 0.0
    for ln in eval_out.splitlines()[::-1]:
        if "Roundtrip accuracy:" in ln:
            try:
                acc = float(ln.split("=")[-1].strip())
            except:
                pass
            break
    print(f"[Round {r}] acc={acc:.4f}")
    if acc >= TARGET:
        print("Target reached. Stop.")
        break
    if not os.path.exists(hard) or os.path.getsize(hard)==0:
        print("No hard examples. Stop.")
        break

    # 组下一轮 mix
    replay = f"{EXP}/replay-r{r}.jsonl"
    mix_next = f"{EXP}/mix-r{r+1}.jsonl"
    sample_replay(DATA, REPLAY_N, replay)
    merge_and_dedup(replay, hard, mix_next)
    mix = mix_next

print("Done.")
