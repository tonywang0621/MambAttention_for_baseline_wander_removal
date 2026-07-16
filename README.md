# MambAttention for ECG Baseline Wander Removal

這個專案保留 `MECG-E-practice` 原本的資料格式、訓練流程與輸出格式，但把模型核心改成接近 `MambAttention` 的架構：

- 輸入資料仍使用 `data/dataset_{noise}_nv{1,2}.pkl`
- 訓練與測試入口仍是 `main.py`
- 輸出仍存成 `results/{config}_{noise}_nv{1,2}.pkl`
- 新模型設定檔是 `config/MambAttention_ECG.yaml`

## 系統需求

建議使用 Linux 或 WSL2。此專案使用 Mamba CUDA extension，實務上需要 NVIDIA GPU。

建議版本：

```text
Python: 3.9 或 3.10
CUDA Toolkit: >= 12.0，建議 12.1 或 12.2
PyTorch: 2.2.2
torchaudio: 2.2.2
```

先確認 GPU 與 CUDA 在目前環境可用：

```bash
nvidia-smi
nvcc -V
```

如果 `nvidia-smi` 或 `nvcc -V` 找不到，請先修好 NVIDIA driver / CUDA Toolkit / WSL GPU 設定，再安裝 Python 套件。

## 建立 Conda 環境

在 Anaconda Prompt、Miniconda shell，或 WSL 裡執行：

```bash
conda create -n mambattention-ecg python=3.10 -y
conda activate mambattention-ecg
```

升級基本安裝工具：

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 安裝 PyTorch

CUDA 12.1 環境建議使用：

```bash
pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

安裝後確認 PyTorch 看得到 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

`torch.cuda.is_available()` 應該要輸出 `True`。

## 安裝 Python 套件

進入專案資料夾：

```bash
cd /mnt/c/Users/中研院/MambAttention_for_baseline_wander_removal/MECG-E-practice
```

安裝 requirements：

```bash
pip install -r requirements.txt
```

## 安裝 Mamba

此專案內含一份 local mamba source，請在同一個 conda 環境中安裝：

```bash
cd mamba
pip install .
cd ..
```

如果安裝 Mamba 時失敗，最常見原因是：

- `nvcc` 不存在或版本太舊
- PyTorch CUDA 版本與本機 CUDA Toolkit 不相容
- 沒有安裝 `ninja`、`packaging`、`wheel`
- 目前環境不是 NVIDIA GPU/CUDA 環境

可以先確認：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc -V
```

## 資料準備

請把資料放在 `data/` 底下。程式會依照 noise type 與 noise version 讀取 pkl：

```text
data/
  dataset_bw_nv1.pkl
  dataset_bw_nv2.pkl
```

如果你使用其他雜訊類型，檔名會依 `--n_type` 改變，例如：

```text
data/dataset_em_nv1.pkl
data/dataset_em_nv2.pkl
data/dataset_ma_nv1.pkl
data/dataset_ma_nv2.pkl
```

每個 pkl 內容必須是：

```python
[X_train, y_train, X_test, y_test]
```

其中：

- `X_train`, `X_test`: 含 baseline wander 的 noisy ECG
- `y_train`, `y_test`: 對應的 clean ECG
- 程式會把資料轉成 PyTorch tensor，並轉為 `[batch, channel, length]`

## 訓練 MambAttention ECG 模型

使用新架構的設定檔：

```bash
python main.py --n_type bw --config config/MambAttention_ECG.yaml
```

這會依序處理：

```text
data/dataset_bw_nv1.pkl
data/dataset_bw_nv2.pkl
```

訓練出的權重會存到：

```text
model_weight/MambAttention_ECG_bw_nv1_weights.pth
model_weight/MambAttention_ECG_bw_nv2_weights.pth
```

測試結果會存到：

```text
results/MambAttention_ECG_bw_nv1.pkl
results/MambAttention_ECG_bw_nv2.pkl
```

結果 pkl 內容仍是：

```python
[X_test, y_test, y_pred]
```

## 只做測試

如果你已經有訓練好的權重，放在 `model_weight/` 底下後執行：

```bash
python main.py --n_type bw --config config/MambAttention_ECG.yaml --test
```

程式會讀取：

```text
model_weight/MambAttention_ECG_bw_nv1_weights.pth
model_weight/MambAttention_ECG_bw_nv2_weights.pth
```

並輸出：

```text
results/MambAttention_ECG_bw_nv1.pkl
results/MambAttention_ECG_bw_nv2.pkl
```

## 計算指標

產生 results 後可用：

```bash
python cal_metrics.py --experiments MambAttention_ECG
```

## 重要設定

主要設定在 `config/MambAttention_ECG.yaml`：

```yaml
train:
  epochs: 30
  batch_size: 96
  lr: 1.0e-4

model:
  fea: "pha"
  dense_channel: 64
  num_tscblocks: 4
  attention_heads: 8
  d_state: 16
  d_conv: 4
  expand: 4
  n_fft: 64
  hop_size: 8
  win_size: 64
  loss_fn: "time+com+con"
```

如果 GPU 記憶體不足，優先降低：

```yaml
train:
  batch_size: 48
```

或降低：

```yaml
model:
  dense_channel: 32
```

## 專案結構

```text
MECG-E-practice/
  main.py
  pipeline.py
  models/
    MECGE.py
  config/
    MambAttention_ECG.yaml
  data/
    dataset_bw_nv1.pkl
    dataset_bw_nv2.pkl
  model_weight/
  results/
  mamba/
  requirements.txt
```
