# Cosmos-Predict2.5 自定义流水线 — 提交代码说明

本目录为工程中与 **LoRA 微调**、**双流动作分支（Step3）**、**数据与嵌入准备** 直接相关的源码汇总。完整运行仍依赖 NVIDIA **Cosmos-Predict2.5** 官方仓库（本机路径通常为与工程根目录并列的 `cosmos-predict2.5-1.5.0`）。

### 架构关系（避免与「只训不推」混淆）

- **视频 LoRA**：`train.py`、Step1/批量推理脚本，管 **Video2World + LoRA**。
- **Step3 双流（本仓库自研）** 刻意拆成 **训练** 与 **推理** 两个入口，职责不同、不能互相替代：
  - **`step3_dual_stream_train.py`**：冻结（或跳过）视频主干，只训练 `VisualBridge` + `ActionDenoiser`，写 `final_model.pt` / `checkpoint_step*.pt`。
  - **`step3_dual_stream_infer.py`**：加载 Step3 权重 +（可选）Cosmos tokenizer 得到 `z0` → `VisualBridge` → **`sample_actions_edm`**（`dual_stream/action_sampling.py`）多步采样，输出 **`action_infer.txt`**。训练脚本里不走这条采样环，因此 **Step3 动作推理必须用这个脚本**，不是遗漏。

---

## 1. 目录结构

```
code/
├── README.md                 # 本说明
├── requirements.txt          # 环境依赖说明（含 Cosmos 安装指引）
├── model/
│   ├── dual_stream/          # 双流模块：视觉桥接、动作 U-Net、数据集、EDM 采样（含 action_sampling.py）
│   ├── scripts/              # 训练/推理入口（需复制到工程根目录 `scripts/`）
│   │   ├── step3_dual_stream_train.py   # Step3 训练
│   │   ├── step3_dual_stream_infer.py   # Step3 动作推理（与 train 配对，必需）
│   │   ├── step4_dual_stream_inference.py # 若存在的后续联合推理实验
│   │   ├── train.py / step1_inference_test.py / batch_lora_test_inference.py …
│   │   └── …
│   └── cosmos_predict2/      # 对官方包的补丁：自定义 LoRA 实验注册
└── data_generation/           # 数据集准备、Reason 向量预计算等
```

---

## 2. 部署到可运行工程（必做）

在包含 `cosmos-predict2.5-1.5.0` 与（可选）`datasets/`、`AIGC/` 的**工程根目录**（例如 `d:\ZTE`）下：

| 本包路径 | 复制到工程根目录 |
|----------|------------------|
| `model/dual_stream/` | `src/dual_stream/`（覆盖或新建 `src`） |
| `model/scripts/*.py` | `scripts/` |
| `model/cosmos_predict2/experiments/base/custom_lora.py` | `cosmos-predict2.5-1.5.0/cosmos_predict2/experiments/base/custom_lora.py` |
| `model/cosmos_predict2/experiments/base/__init__.py` | 同上（若你本地已合并 `custom_lora` 导入则可跳过） |

脚本假定：`scripts/*.py` 与 `src/`、`cosmos-predict2.5-1.5.0/` 位于**同一工程根目录**，与现有仓库布局一致。

---

## 3. 环境配置

详见同目录 **`requirements.txt`**。摘要：

- **Python**：建议 3.10（与官方 `pyproject.toml` 一致）。
- **Cosmos 依赖**：在 `cosmos-predict2.5-1.5.0` 目录内按官方方式安装，例如 `uv sync --extra cu128` 或 `cu130`，再激活该环境的 Python。
- **本仓库脚本**：在已激活的 Cosmos 虚拟环境中运行，以便 `import cosmos_predict2`、`torch` 等。

---

## 4. 典型命令（在工程根目录执行）

### 4.1 预计算 Reason 文本嵌入（训练侧 `compute_online=False` 时使用）

```bash
python scripts/get_reason_embeddings.py --dataset_dir datasets/custom_lora
```

### 4.2 LoRA 微调训练

在 `cosmos-predict2.5-1.5.0` 目录内：

```bash
torchrun --nproc_per_node=1 --master_port=12376 scripts/train.py \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- \
  experiment=predict2_video2world_training_2b_custom_lora \
  trainer.max_iter=1000 \
  checkpoint.save_iter=500 \
  job.name=custom_lora_run \
  job.wandb_mode=disabled
```

或从工程根使用包装脚本（内部会切换到 Cosmos 根目录）：

```bash
./cosmos-predict2.5-1.5.0/.venv/bin/python scripts/train.py \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- \
  experiment=predict2_video2world_training_2b_custom_lora
```

实验配置见补丁文件 `custom_lora.py`（数据集路径、帧数、分辨率、`min/max_num_conditional_frames`、`text_encoder_config.compute_online` 等）；可按 Hydra 语法在命令行覆盖。

### 4.3 单条样本推理（LoRA checkpoint）

```bash
cd cosmos-predict2.5-1.5.0
python ../scripts/step1_inference_test.py \
  --ckpt_path /path/to/checkpoints/iter_000001000 \
  --input_video ../AIGC/release/train/1_1/video.mp4 \
  --instruction_file ../AIGC/release/train/1_1/instruction.txt \
  --output_dir ../outputs/step1_lora \
  --resolution 352,640 \
  --guidance 3
```

### 4.4 批量测试集推理（推荐；支持较大条件帧数 K）

```bash
cd cosmos-predict2.5-1.5.0
python ../scripts/batch_lora_test_inference.py \
  --ckpt_path /path/to/checkpoints/iter_000001000 \
  --test_root /path/to/AIGC/release/test \
  --output_dir /path/to/outputs/lora_infer \
  --num_latent_conditional_frames 4 \
  --preprocess auto \
  --guidance 3 \
  --height 352 --width 640 \
  --pad_short_condition
```

### 4.5 Step3 双流动作分支训练

```bash
python scripts/step3_dual_stream_train.py \
  --ckpt_path /path/to/Cosmos-Predict2.5-2B \
  --data_root AIGC/release/train \
  --embedding_dir outputs/text_embeddings \
  --output_dir outputs/step3_dual_stream \
  --batch_size 4 \
  --max_steps 10000 \
  --learning_rate 1e-4
```

### 4.6 Step3 动作推理（EDM 多步采样）

```bash
python scripts/step3_dual_stream_infer.py \
  --step3_ckpt outputs/step3_dual_stream/final_model.pt \
  --data_root AIGC/release/train \
  --input_root AIGC/release/test \
  --output_dir outputs/step3_action_infer \
  --cosmos_ckpt nvidia/Cosmos-Predict2.5-2B \
  --embedding_dir outputs/text_embeddings \
  --num_sampling_steps 30
```

---

## 5. 数据与配置约定

- **LoRA 数据**：默认 `datasets/custom_lora/`（与 `VideoDataset` 的 `dataset_dir` 一致）；赛题样本可参考 `AIGC/release/train`、`test`（`video.mp4`、`instruction.txt` 等）。
- **补丁中的路径**：`custom_lora.py` 内 `dataset_dir` 等可能保留服务器路径（如 `/root/autodl-tmp/...`），部署到本机后请改为实际路径或通过 Hydra 覆盖。
- **双流模块**：`src/dual_stream/action_sampling.py` 为 Step3 推理多步 EDM 采样，与训练循环中的 EDM 预条件化一致。

---

## 6. 更多文档

工程根目录下的 **`PIPELINE_GUIDE.md`**、**`step3_dual_stream_technical_summary.md`** 中有故障排查、参数含义与阶段说明，可与本 README 对照阅读（未纳入本 `code/` 包时请从原仓库一并查阅）。
