python data_prep.py --input data/my_dataset.json --output new_data.jsonl --dedup --arch aarch64

python train_lora.py --train_file new_data.jsonl --output_dir experiments/Qwen --base_model models/Qwen2.5-Coder-3B-Instruct --epochs 1 --batch 1 --grad_accum 16 --lr 2e-4

python infer_once.py --adapter experiments\Qwen --base_model models\Qwen2.5-Coder-3B-Instruct --offload_dir offload  --context "sub sp, sp, #16`nstr wzr, [sp, #12]" --query   "mov w8, #1"

python eval_roundtrip.py `
  --base_model Qwen/Qwen2.5-Coder-3B-Instruct `
  --adapter experiments/Qwen `
  --data data\my_data.jsonl `
  --save_hard hard_examples.jsonl
  -- offload_dir offload

# 0) 准备目录
sudo mkdir -p /srv/{env,models,experiments,offload,data,hf-cache}
sudo chown -R $USER /srv
export HF_HOME=/srv/hf-cache
export TRANSFORMERS_CACHE=/srv/hf-cache

# 1) 新建 conda 环境
conda create -n mc2mi python=3.10 -y
conda activate mc2mi

# 2) 安装 PyTorch (CUDA 12.1) —— 与你本地一致
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3) 安装训练/推理依赖
pip install -U transformers accelerate peft bitsandbytes datasets sentencepiece evaluate regex huggingface_hub hf-transfer

# 4) （可选）按你的 requirements.txt 精调版本
# 把 requirements.txt 上传到 /srv/env/ 后：
pip install -r /srv/env/requirements.txt || true

# 5) （可选）还原 accelerate 配置
# 如果你导出了 accelerate.yaml，放到：~/.cache/huggingface/accelerate/default_config.yaml
mkdir -p ~/.cache/huggingface/accelerate
cp /srv/env/accelerate.yaml ~/.cache/huggingface/accelerate/default_config.yaml 2>/dev/null || true




# 1) 数据清洗
python data_prep.py \
  --input /srv/data/my_dataset.json \
  --output /srv/data/new_data.jsonl \
  --dedup --arch aarch64

# 2) 训练（A100 推荐直接用本地路径）
python train_lora.py \
  --train_file all_data.jsonl \
  --output_dir out/qwen2_5_mi_lora_bf16 \
  --base_model models/Qwen2.5-Coder-7B \
  --epochs 1 --batch 1 --grad_accum 16 --lr 2e-4

# 3) 单条推理（A100可不写 offload，但保留不影响）
python infer_once.py \
  --adapter out/qwen2_5_mi_lora_bf16 \
  --base_model models/Qwen2.5-Coder-7B \
  --offload_dir offload \
  --context "sub sp, sp, #16\nstr wzr, [sp, #12]" \
  --query   "mov w8, #1"

# 4) 批量验证
python eval_roundtrip.py \
  --base_model /srv/models/Qwen2.5-Coder-7B \
  --adapter    /srv/experiments/Qwen \
  --data       /srv/data/new_data.jsonl \
  --save_hard  /srv/experiments/hard_examples.jsonl \
  --offload_dir /srv/offload


# 先下载
huggingface-cli download Qwen/Qwen2.5-Coder-7B \
  --local-dir /srv/models/Qwen2.5-Coder-7B

# 训练命令只改这一个参数即可
python train_lora.py \
  --train_file /srv/data/new_data.jsonl \
  --output_dir /srv/experiments/Qwen-7B \
  --base_model /srv/models/Qwen2.5-Coder-7B \
  --epochs 1 --batch 2 --grad_accum 16 --lr 2e-4
