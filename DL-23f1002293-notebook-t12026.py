{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3d842351",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T16:00:20.498895Z",
     "iopub.status.busy": "2026-03-12T16:00:20.498614Z",
     "iopub.status.idle": "2026-03-12T16:00:25.041406Z",
     "shell.execute_reply": "2026-03-12T16:00:25.040470Z"
    },
    "papermill": {
     "duration": 4.549707,
     "end_time": "2026-03-12T16:00:25.042939",
     "exception": false,
     "start_time": "2026-03-12T16:00:20.493232",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting pyloudnorm\r\n",
      "  Downloading pyloudnorm-0.2.0-py3-none-any.whl.metadata (6.6 kB)\r\n",
      "Requirement already satisfied: scipy>=1.0.1 in /usr/local/lib/python3.12/dist-packages (from pyloudnorm) (1.16.3)\r\n",
      "Requirement already satisfied: numpy>=1.14.2 in /usr/local/lib/python3.12/dist-packages (from pyloudnorm) (2.0.2)\r\n",
      "Downloading pyloudnorm-0.2.0-py3-none-any.whl (10 kB)\r\n",
      "Installing collected packages: pyloudnorm\r\n",
      "Successfully installed pyloudnorm-0.2.0\r\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "%pip install pyloudnorm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b488e4af",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T16:00:25.049342Z",
     "iopub.status.busy": "2026-03-12T16:00:25.048865Z",
     "iopub.status.idle": "2026-03-12T16:00:41.464778Z",
     "shell.execute_reply": "2026-03-12T16:00:41.464150Z"
    },
    "papermill": {
     "duration": 16.421024,
     "end_time": "2026-03-12T16:00:41.466636",
     "exception": false,
     "start_time": "2026-03-12T16:00:25.045612",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Importing Libraries \n",
    "\n",
    "import os\n",
    "import glob\n",
    "import json\n",
    "import random\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import librosa\n",
    "import soundfile as sf\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "import torchaudio\n",
    "import torchaudio.transforms as T\n",
    "import timm\n",
    "from scipy.signal import butter, sosfilt\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from tqdm import tqdm\n",
    "from sklearn.metrics import f1_score\n",
    "import pyloudnorm as l_norm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "0d1864d0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T16:00:41.472917Z",
     "iopub.status.busy": "2026-03-12T16:00:41.472523Z",
     "iopub.status.idle": "2026-03-12T16:00:41.477440Z",
     "shell.execute_reply": "2026-03-12T16:00:41.476749Z"
    },
    "papermill": {
     "duration": 0.009489,
     "end_time": "2026-03-12T16:00:41.478769",
     "exception": false,
     "start_time": "2026-03-12T16:00:41.469280",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Paths \n",
    "\n",
    "DATA_ROOT = \"/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup\"\n",
    "SRC_STEMS = os.path.join(DATA_ROOT, \"genres_stems\")\n",
    "TEST_CSV = os.path.join(DATA_ROOT, \"test.csv\")\n",
    "TEST_AUDIO = os.path.join(DATA_ROOT, \"mashups\")\n",
    "WORK_DIR = \"/kaggle/working\"\n",
    "TEMP_DIR='/tmp'\n",
    "ESC50_SRC = \"/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/ESC-50-master/audio\"\n",
    "NOISE_BANK_DIR = os.path.join(TEMP_DIR, \"ast_noise_bank\")\n",
    "STATS_PATH = os.path.join(TEMP_DIR, \"dataset_stats.json\")\n",
    "FEATURES_ROOT = os.path.join(TEMP_DIR, \"ast_features\")\n",
    "CHECKPOINT_PATH = os.path.join(WORK_DIR, \"best_ast_multistem.pth\")\n",
    "RESUME_PATH = os.path.join(WORK_DIR, \"latest_ast_multistem.pth\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "63d83846",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T16:00:41.484417Z",
     "iopub.status.busy": "2026-03-12T16:00:41.484173Z",
     "iopub.status.idle": "2026-03-12T16:00:41.737292Z",
     "shell.execute_reply": "2026-03-12T16:00:41.736580Z"
    },
    "papermill": {
     "duration": 0.257719,
     "end_time": "2026-03-12T16:00:41.738880",
     "exception": false,
     "start_time": "2026-03-12T16:00:41.481161",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Configuration\n",
    "\n",
    "SAMPLING_RATE = 16000\n",
    "STEM_NAMES = [\"vocals\", \"drums\", \"bass\", \"other\"]\n",
    "TRIM_DB = 60\n",
    "HPF_CUTOFF = 30.0\n",
    "TARGET_LUFS = -14.0\n",
    "\n",
    "N_MELS = 128\n",
    "WIN_LENGTH = 400 \n",
    "HOP_LENGTH = 160 \n",
    "N_FFT = 1024 \n",
    "TARGET_DURATION_SEC = 10.24 \n",
    "TARGET_SAMPLES = int(TARGET_DURATION_SEC * SAMPLING_RATE)\n",
    "BATCH_SIZE = 4\n",
    "LR = 1e-5\n",
    "EPOCHS = 20\n",
    "\n",
    "DEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"mps\" if torch.backends.mps.is_available() else \"cpu\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "28b12911",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T16:00:41.745561Z",
     "iopub.status.busy": "2026-03-12T16:00:41.745297Z",
     "iopub.status.idle": "2026-03-12T16:00:41.758047Z",
     "shell.execute_reply": "2026-03-12T16:00:41.757487Z"
    },
    "papermill": {
     "duration": 0.017723,
     "end_time": "2026-03-12T16:00:41.759422",
     "exception": false,
     "start_time": "2026-03-12T16:00:41.741699",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Preprocessing functions \n",
    "\n",
    "def get_mel_transform():\n",
    "    return T.MelSpectrogram(\n",
    "        sample_rate=SAMPLING_RATE, n_fft=N_FFT, win_length=WIN_LENGTH,\n",
    "        hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0\n",
    "    )\n",
    "\n",
    "def extract_power_spec(y):\n",
    "    if len(y) > TARGET_SAMPLES:\n",
    "        y = y[:TARGET_SAMPLES]\n",
    "    else:\n",
    "        y = np.pad(y, (0, max(0, TARGET_SAMPLES - len(y))))\n",
    "\n",
    "    tensor_y = torch.from_numpy(y).float()\n",
    "    transform = get_mel_transform()\n",
    "    spec = transform(tensor_y)\n",
    "    return spec[:, :1024].T # (1024, 128)\n",
    "\n",
    "def process_one_song(song_id, genre_name):\n",
    "    target_dir = os.path.join(FEATURES_ROOT, genre_name, song_id)\n",
    "    if os.path.exists(target_dir):\n",
    "        existing = [f for f in os.listdir(target_dir) if f.endswith('.pt')]\n",
    "        if len(existing) >= len(STEM_NAMES): return True\n",
    "\n",
    "    try:\n",
    "        y_stems = {}\n",
    "        os.makedirs(target_dir, exist_ok=True)\n",
    "        for st in STEM_NAMES:\n",
    "            p = os.path.join(SRC_STEMS, genre_name, song_id, f\"{st}.wav\")\n",
    "            if not os.path.exists(p): return False\n",
    "            y, _ = librosa.load(p, sr=SAMPLING_RATE)\n",
    "            y_stems[st] = y\n",
    "\n",
    "        full_mix = np.sum(list(y_stems.values()), axis=0)\n",
    "        _, trim_idx = librosa.effects.trim(full_mix, top_db=TRIM_DB)\n",
    "        start, end = trim_idx\n",
    "\n",
    "        nyquist = 0.5 * SAMPLING_RATE\n",
    "        sos = butter(2, HPF_CUTOFF/nyquist, btype='high', output='sos')\n",
    "        meter = l_norm.Meter(SAMPLING_RATE)\n",
    "\n",
    "        normalized_stems = {name: sosfilt(sos, y[start:end]) for name, y in y_stems.items()}\n",
    "        mix_proc = np.sum(list(normalized_stems.values()), axis=0)\n",
    "        loudness = meter.integrated_loudness(mix_proc.reshape(-1, 1))\n",
    "\n",
    "        delta_loudness = TARGET_LUFS - loudness\n",
    "        gain = 10.0 ** (delta_loudness / 20.0)\n",
    "\n",
    "        for name, y in normalized_stems.items():\n",
    "            y_norm = y * gain\n",
    "            peak = np.max(np.abs(y_norm))\n",
    "            if peak > 0.89: y_norm = y_norm * (0.89 / peak) \n",
    "            \n",
    "            spec = extract_power_spec(y_norm)\n",
    "            torch.save(spec, os.path.join(target_dir, f\"{name}_spec.pt\"))\n",
    "        return True\n",
    "    except Exception as e:\n",
    "        print(f\"Error processing {song_id}: {e}\")\n",
    "        return False\n",
    "\n",
    "def process_noise_bank():\n",
    "    if not os.path.exists(ESC50_SRC): return\n",
    "    os.makedirs(NOISE_BANK_DIR, exist_ok=True)\n",
    "    noise_files = glob.glob(os.path.join(ESC50_SRC, \"*.wav\"))\n",
    "    for f in tqdm(noise_files, desc=\"Processing Noise Bank\"):\n",
    "        out_path = os.path.join(NOISE_BANK_DIR, os.path.basename(f).replace(\".wav\", \".pt\"))\n",
    "        if os.path.exists(out_path): continue\n",
    "        try:\n",
    "            y, _ = librosa.load(f, sr=SAMPLING_RATE)\n",
    "            torch.save(extract_power_spec(y), out_path)\n",
    "        except: continue\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "e9d62085",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T16:00:41.765242Z",
     "iopub.status.busy": "2026-03-12T16:00:41.764981Z",
     "iopub.status.idle": "2026-03-12T16:00:41.772664Z",
     "shell.execute_reply": "2026-03-12T16:00:41.772079Z"
    },
    "papermill": {
     "duration": 0.0123,
     "end_time": "2026-03-12T16:00:41.773975",
     "exception": false,
     "start_time": "2026-03-12T16:00:41.761675",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Model \n",
    "\n",
    "class ASTMultiStemModel(nn.Module):\n",
    "    def __init__(self, num_classes=10):\n",
    "        super().__init__()\n",
    "        self.encoders = nn.ModuleList([self._create_encoder() for _ in range(4)])\n",
    "        self.fc = nn.Linear(192 * 4, num_classes)\n",
    "\n",
    "    def _create_encoder(self):\n",
    "        m = timm.create_model('deit_tiny_patch16_224', pretrained=True)\n",
    "        m.patch_embed.img_size = (1024, 128)\n",
    "        m.patch_embed.grid_size = (101, 12)\n",
    "        m.patch_embed.num_patches = 101 * 12\n",
    "        \n",
    "        old_proj = m.patch_embed.proj\n",
    "        m.patch_embed.proj = nn.Conv2d(1, 192, kernel_size=(16, 16), stride=(10, 10))\n",
    "        with torch.no_grad():\n",
    "            m.patch_embed.proj.weight.copy_(old_proj.weight.mean(dim=1, keepdim=True))\n",
    "            m.patch_embed.proj.bias.copy_(old_proj.bias)\n",
    "            cls_pos = m.pos_embed[:, :1, :]\n",
    "            patch_pos = m.pos_embed[:, 1:, :].reshape(1, 14, 14, 192).permute(0, 3, 1, 2)\n",
    "            new_patch_pos = nn.functional.interpolate(patch_pos, size=(101, 12), mode='bicubic')\n",
    "            new_patch_pos = new_patch_pos.permute(0, 2, 3, 1).reshape(1, 101*12, 192)\n",
    "            m.pos_embed = nn.Parameter(torch.cat([cls_pos, new_patch_pos], dim=1))\n",
    "        m.head = nn.Identity()\n",
    "        return m\n",
    "\n",
    "    def forward(self, x):\n",
    "        cls_tokens = [self.encoders[i](x[:, i, :, :].unsqueeze(1)) for i in range(4)]\n",
    "        return self.fc(torch.cat(cls_tokens, dim=1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "e45de1ad",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T16:00:41.779597Z",
     "iopub.status.busy": "2026-03-12T16:00:41.779358Z",
     "iopub.status.idle": "2026-03-12T16:00:41.784622Z",
     "shell.execute_reply": "2026-03-12T16:00:41.784054Z"
    },
    "papermill": {
     "duration": 0.009904,
     "end_time": "2026-03-12T16:00:41.786188",
     "exception": false,
     "start_time": "2026-03-12T16:00:41.776284",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def calculate_stats():\n",
    "    all_specs = glob.glob(os.path.join(FEATURES_ROOT, \"**/vocals_spec.pt\"), recursive=True)[:200]\n",
    "    if not all_specs: raise RuntimeError(\"No features found\")\n",
    "    \n",
    "    sum_v, sum_sq, count = 0.0, 0.0, 0\n",
    "    for p in tqdm(all_specs, desc=\"Stats\"):\n",
    "        spec = torch.log(torch.load(p) + 1e-10)\n",
    "        sum_v += spec.sum().item(); sum_sq += (spec**2).sum().item(); count += spec.numel()\n",
    "    \n",
    "    mean = sum_v / count\n",
    "    std = ((sum_sq / count) - (mean**2))**0.5\n",
    "    stats = {'dataset_mean': mean, 'dataset_std': std}\n",
    "    with open(STATS_PATH, 'w') as f: json.dump(stats, f)\n",
    "    return stats"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d08ab8eb",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T16:00:41.791896Z",
     "iopub.status.busy": "2026-03-12T16:00:41.791668Z",
     "iopub.status.idle": "2026-03-12T16:00:41.799660Z",
     "shell.execute_reply": "2026-03-12T16:00:41.799125Z"
    },
    "papermill": {
     "duration": 0.012382,
     "end_time": "2026-03-12T16:00:41.800884",
     "exception": false,
     "start_time": "2026-03-12T16:00:41.788502",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Dataset \n",
    "\n",
    "class HybridAudioDataset(Dataset):\n",
    "    def __init__(self, stats, is_train=True):\n",
    "        self.is_train, self.mean, self.std = is_train, stats['dataset_mean'], stats['dataset_std']\n",
    "        self.samples = []\n",
    "        genres = sorted([d for d in os.listdir(FEATURES_ROOT) if not d.startswith('.') and os.path.isdir(os.path.join(FEATURES_ROOT, d))])\n",
    "        self.g2i = {g: i for i, g in enumerate(genres)}\n",
    "        for g in genres:\n",
    "            g_dir = os.path.join(FEATURES_ROOT, g)\n",
    "            for sid in os.listdir(g_dir):\n",
    "                if not sid.startswith('.'): self.samples.append({'path': os.path.join(g_dir, sid), 'label': self.g2i[g]})\n",
    "        self.noises = glob.glob(os.path.join(NOISE_BANK_DIR, \"*.pt\"))\n",
    "        self.freq_mask, self.time_mask = T.FrequencyMasking(30), T.TimeMasking(120)\n",
    "\n",
    "    def __len__(self): return len(self.samples)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        s = self.samples[idx]\n",
    "        stem_specs = [torch.load(os.path.join(s['path'], f\"{st}_spec.pt\")) for st in STEM_NAMES]\n",
    "        processed = []\n",
    "        for spec in stem_specs:\n",
    "            if self.is_train and self.noises and random.random() < 0.5:\n",
    "                n_spec = torch.load(random.choice(self.noises))\n",
    "                scale = (spec.mean() / (n_spec.mean() * (10**1.5)))**0.5 if n_spec.mean() > 0 else 1.0\n",
    "                spec = spec + (n_spec * scale)\n",
    "            spec = torch.log(spec + 1e-10)\n",
    "            if self.is_train: spec = self.time_mask(self.freq_mask(spec.T)).T\n",
    "            processed.append((spec - self.mean) / self.std)\n",
    "        return torch.stack(processed), s['label']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "205d0e91",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T16:00:41.806311Z",
     "iopub.status.busy": "2026-03-12T16:00:41.806055Z",
     "iopub.status.idle": "2026-03-12T16:00:41.811679Z",
     "shell.execute_reply": "2026-03-12T16:00:41.811139Z"
    },
    "papermill": {
     "duration": 0.009861,
     "end_time": "2026-03-12T16:00:41.812959",
     "exception": false,
     "start_time": "2026-03-12T16:00:41.803098",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Training\n",
    "\n",
    "def preprocessing():\n",
    "    os.makedirs(FEATURES_ROOT, exist_ok=True)\n",
    "    process_noise_bank()\n",
    "    \n",
    "    tasks = []\n",
    "    genres = [g for g in sorted(os.listdir(SRC_STEMS)) if not g.startswith('.')]\n",
    "    for g in genres:\n",
    "        g_path = os.path.join(SRC_STEMS, g)\n",
    "        if os.path.isdir(g_path):\n",
    "            sids = [s for s in os.listdir(g_path) if not s.startswith('.')]\n",
    "            for sid in sids: tasks.append((sid, g))\n",
    "\n",
    "    for sid, g in tqdm(tasks, desc=\"Extracting Features\"):\n",
    "        process_one_song(sid, g)\n",
    "\n",
    "    if not os.path.exists(STATS_PATH):\n",
    "        stats = calculate_stats()\n",
    "    else:\n",
    "        with open(STATS_PATH, 'r') as f: stats = json.load(f)\n",
    "    print(f\"Pipeline Ready. Mean: {stats['dataset_mean']:.4f}\")\n",
    "    return stats\n",
    "\n",
    "def get_stats():\n",
    "    stats = json.load(open(STATS_PATH))\n",
    "    return stats\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "40b24fc3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T16:00:41.818751Z",
     "iopub.status.busy": "2026-03-12T16:00:41.818521Z",
     "iopub.status.idle": "2026-03-12T16:00:41.827436Z",
     "shell.execute_reply": "2026-03-12T16:00:41.826743Z"
    },
    "papermill": {
     "duration": 0.0135,
     "end_time": "2026-03-12T16:00:41.828761",
     "exception": false,
     "start_time": "2026-03-12T16:00:41.815261",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def training(stats):\n",
    "    full_ds = HybridAudioDataset(stats, is_train=False)\n",
    "    indices = list(range(len(full_ds))); random.shuffle(indices)\n",
    "    t_size = int(0.9 * len(full_ds))\n",
    "    t_ds = torch.utils.data.Subset(HybridAudioDataset(stats, is_train=True), indices[:t_size])\n",
    "    v_ds = torch.utils.data.Subset(full_ds, indices[t_size:])\n",
    "\n",
    "    t_loader = DataLoader(t_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)\n",
    "    v_loader = DataLoader(v_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)\n",
    "\n",
    "    model = ASTMultiStemModel(num_classes=10).to(DEVICE)\n",
    "    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-7)\n",
    "    criterion = nn.CrossEntropyLoss()\n",
    "    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)\n",
    "\n",
    "    start_epoch, best_f1 = 0, 0\n",
    "    if os.path.exists(RESUME_PATH):\n",
    "        ckpt = torch.load(RESUME_PATH, map_location=DEVICE)\n",
    "        model.load_state_dict(ckpt['model_state_dict'])\n",
    "        optimizer.load_state_dict(ckpt['optimizer_state_dict'])\n",
    "        scheduler.load_state_dict(ckpt['scheduler_state_dict'])\n",
    "        start_epoch, best_f1 = ckpt['epoch'] + 1, ckpt.get('best_f1', 0)\n",
    "\n",
    "    for epoch in range(start_epoch, EPOCHS):\n",
    "        model.train()\n",
    "        for imgs, labels in tqdm(t_loader, desc=f\"Epoch {epoch+1}\"):\n",
    "            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)\n",
    "            optimizer.zero_grad(); criterion(model(imgs), labels).backward(); optimizer.step()\n",
    "        scheduler.step()\n",
    "\n",
    "        model.eval(); all_p, all_l = [], []\n",
    "        with torch.no_grad():\n",
    "            for imgs, labels in v_loader:\n",
    "                logits = model(imgs.to(DEVICE))\n",
    "                all_p.extend(logits.max(1)[1].cpu().numpy()); all_l.extend(labels.numpy())\n",
    "        f1 = f1_score(all_l, all_p, average='macro')\n",
    "        print(f\"Val F1: {f1:.4f}\")\n",
    "\n",
    "        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_f1': max(f1, best_f1)}, RESUME_PATH)\n",
    "        if f1 > best_f1:\n",
    "            best_f1 = f1\n",
    "            torch.save(model.state_dict(), CHECKPOINT_PATH)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "f7c06769",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T16:00:41.834373Z",
     "iopub.status.busy": "2026-03-12T16:00:41.834133Z",
     "iopub.status.idle": "2026-03-12T16:00:41.842717Z",
     "shell.execute_reply": "2026-03-12T16:00:41.842017Z"
    },
    "papermill": {
     "duration": 0.013003,
     "end_time": "2026-03-12T16:00:41.844056",
     "exception": false,
     "start_time": "2026-03-12T16:00:41.831053",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Inference \n",
    "\n",
    "def inference(stats):\n",
    "    test_df = pd.read_csv(TEST_CSV)\n",
    "    model = ASTMultiStemModel(num_classes=10).to(DEVICE)\n",
    "    \n",
    "    if not os.path.exists(CHECKPOINT_PATH):\n",
    "        print(f\"No checkpoint found at {CHECKPOINT_PATH}\")\n",
    "        return\n",
    "        \n",
    "    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))\n",
    "    model.eval()\n",
    "\n",
    "    mean, std = stats['dataset_mean'], stats['dataset_std']\n",
    "    genres = sorted([d for d in os.listdir(FEATURES_ROOT) if not d.startswith('.') and os.path.isdir(os.path.join(FEATURES_ROOT, d))])\n",
    "    preds = []\n",
    "    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=\"Inference\"):\n",
    "        sid = str(row['id']).zfill(4)\n",
    "        mashup_path = os.path.join(DATA_ROOT, row['filename'])\n",
    "        \n",
    "        tta_logits = []\n",
    "        for shift in [0, 800, 1600]: \n",
    "            specs_shift = []\n",
    "            for st in STEM_NAMES:\n",
    "                p = os.path.join(TEST_AUDIO, sid, f\"{st}.wav\")\n",
    "                if not os.path.exists(p):\n",
    "                    p = os.path.join(TEST_AUDIO, f\"song{sid}\", f\"{st}.wav\")\n",
    "                \n",
    "                audio_source = p if os.path.exists(p) else mashup_path\n",
    "                \n",
    "                try:\n",
    "                    y, _ = librosa.load(audio_source, sr=SAMPLING_RATE)\n",
    "                    if shift > 0: y = np.roll(y, shift)\n",
    "                    spec = extract_power_spec(y)\n",
    "                    specs_shift.append((torch.log(spec + 1e-10) - mean) / std)\n",
    "                except Exception as e:\n",
    "                    print(f\"Error loading {audio_source}: {e}\")\n",
    "                    specs_shift.append(torch.zeros(1024, 128))\n",
    "            \n",
    "            inp = torch.stack(specs_shift).unsqueeze(0).to(DEVICE)\n",
    "            with torch.no_grad():\n",
    "                tta_logits.append(model(inp))\n",
    "        \n",
    "        avg_logits = torch.stack(tta_logits).mean(0)\n",
    "        res = avg_logits.max(1)[1].item()\n",
    "        preds.append(genres[res])\n",
    "            \n",
    "    test_df['genre'] = preds\n",
    "    test_df[['id', 'genre']].to_csv(\"submission.csv\", index=False)\n",
    "    print(\"Done!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "55aa5a3a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T16:00:41.849516Z",
     "iopub.status.busy": "2026-03-12T16:00:41.849271Z",
     "iopub.status.idle": "2026-03-12T17:23:20.735590Z",
     "shell.execute_reply": "2026-03-12T17:23:20.734695Z"
    },
    "papermill": {
     "duration": 4958.890713,
     "end_time": "2026-03-12T17:23:20.737028",
     "exception": false,
     "start_time": "2026-03-12T16:00:41.846315",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running Pipeline\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing Noise Bank: 100%|██████████| 2000/2000 [01:07<00:00, 29.72it/s]\n",
      "Extracting Features: 100%|██████████| 1000/1000 [09:29<00:00,  1.76it/s]\n",
      "Stats: 100%|██████████| 200/200 [00:00<00:00, 983.13it/s] \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Pipeline Ready. Mean: -8.0320\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "47a77355b1ca4db7a24465c3d620a728",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "model.safetensors:   0%|          | 0.00/22.9M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n",
      "Epoch 1: 100%|██████████| 225/225 [02:24<00:00,  1.56it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Val F1: 0.5990\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 2: 100%|██████████| 225/225 [02:31<00:00,  1.48it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Val F1: 0.5820\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 3: 100%|██████████| 225/225 [02:35<00:00,  1.45it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Val F1: 0.7146\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 4: 100%|██████████| 225/225 [02:36<00:00,  1.44it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Val F1: 0.6950\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 5: 100%|██████████| 225/225 [02:37<00:00,  1.43it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Val F1: 0.7726\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 6: 100%|██████████| 225/225 [02:37<00:00,  1.43it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Val F1: 0.7396\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 7: 100%|██████████| 225/225 [02:36<00:00,  1.44it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Val F1: 0.8102\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 8: 100%|██████████| 225/225 [02:37<00:00,  1.43it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Val F1: 0.8159\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 9: 100%|██████████| 225/225 [02:36<00:00,  1.43it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Val F1: 0.7842\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 10: 100%|██████████| 225/225 [02:37<00:00,  1.43it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Val F1: 0.8120\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 11: 100%|██████████| 225/225 [02:36<00:00,  1.43it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Val F1: 0.8294\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 12: 100%|██████████| 225/225 [02:37<00:00,  1.43it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Val F1: 0.8075\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 13: 100%|██████████| 225/225 [02:36<00:00,  1.43it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Val F1: 0.7990\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 14: 100%|██████████| 225/225 [02:37<00:00,  1.43it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Val F1: 0.8370\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 15: 100%|██████████| 225/225 [02:36<00:00,  1.43it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Val F1: 0.7933\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 16: 100%|██████████| 225/225 [02:36<00:00,  1.43it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Val F1: 0.7903\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 17: 100%|██████████| 225/225 [02:37<00:00,  1.43it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Val F1: 0.8098\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 18: 100%|██████████| 225/225 [02:37<00:00,  1.43it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Val F1: 0.8242\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 19: 100%|██████████| 225/225 [02:37<00:00,  1.43it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Val F1: 0.8385\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 20: 100%|██████████| 225/225 [02:37<00:00,  1.43it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Val F1: 0.8385\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Inference: 100%|██████████| 3020/3020 [18:02<00:00,  2.79it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Done!\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "if __name__ == \"__main__\":\n",
    "    print(f\"Running Pipeline\")\n",
    "\n",
    "    preprocessing()\n",
    "    training(get_stats())\n",
    "    inference(get_stats())"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "nvidiaTeslaT4",
   "dataSources": [
    {
     "databundleVersionId": 15477148,
     "sourceId": 128431,
     "sourceType": "competition"
    }
   ],
   "isGpuEnabled": true,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 4986.43695,
   "end_time": "2026-03-12T17:23:24.284618",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-03-12T16:00:17.847668",
   "version": "2.6.0"
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "state": {
     "0ca15b8b976a468388dfdce74af186fb": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "2.0.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "2.0.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border_bottom": null,
       "border_left": null,
       "border_right": null,
       "border_top": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "47a77355b1ca4db7a24465c3d620a728": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "HBoxModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "HBoxModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "2.0.0",
       "_view_name": "HBoxView",
       "box_style": "",
       "children": [
        "IPY_MODEL_5af04b9d9ebe490d86c4ab4de2d1bdfb",
        "IPY_MODEL_acb540d8512646f0815337f099ef74be",
        "IPY_MODEL_e8012b588a8047beadea535779e14876"
       ],
       "layout": "IPY_MODEL_0ca15b8b976a468388dfdce74af186fb",
       "tabbable": null,
       "tooltip": null
      }
     },
     "5af04b9d9ebe490d86c4ab4de2d1bdfb": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "2.0.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_allow_html": false,
       "layout": "IPY_MODEL_9c63b53180884714a3954592f6e0e1a0",
       "placeholder": "​",
       "style": "IPY_MODEL_ab4da44d7be94f4b9449f3809d009153",
       "tabbable": null,
       "tooltip": null,
       "value": "model.safetensors: 100%"
      }
     },
     "976749d4dbd84610aca6ff554c2b870b": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "2.0.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "2.0.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border_bottom": null,
       "border_left": null,
       "border_right": null,
       "border_top": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "9be19991577a4271ab5b11918ddd48d0": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "HTMLStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "HTMLStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "StyleView",
       "background": null,
       "description_width": "",
       "font_size": null,
       "text_color": null
      }
     },
     "9c63b53180884714a3954592f6e0e1a0": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "2.0.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "2.0.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border_bottom": null,
       "border_left": null,
       "border_right": null,
       "border_top": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "9f33db09afa34cdf83ce63da37e536ce": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "2.0.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "2.0.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border_bottom": null,
       "border_left": null,
       "border_right": null,
       "border_top": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "ab4da44d7be94f4b9449f3809d009153": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "HTMLStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "HTMLStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "StyleView",
       "background": null,
       "description_width": "",
       "font_size": null,
       "text_color": null
      }
     },
     "acb540d8512646f0815337f099ef74be": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "FloatProgressModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "FloatProgressModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "2.0.0",
       "_view_name": "ProgressView",
       "bar_style": "success",
       "description": "",
       "description_allow_html": false,
       "layout": "IPY_MODEL_9f33db09afa34cdf83ce63da37e536ce",
       "max": 22883348.0,
       "min": 0.0,
       "orientation": "horizontal",
       "style": "IPY_MODEL_b89f775bd5e74d02b4368dbae23f9bad",
       "tabbable": null,
       "tooltip": null,
       "value": 22883348.0
      }
     },
     "b89f775bd5e74d02b4368dbae23f9bad": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "ProgressStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "StyleView",
       "bar_color": null,
       "description_width": ""
      }
     },
     "e8012b588a8047beadea535779e14876": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "2.0.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_allow_html": false,
       "layout": "IPY_MODEL_976749d4dbd84610aca6ff554c2b870b",
       "placeholder": "​",
       "style": "IPY_MODEL_9be19991577a4271ab5b11918ddd48d0",
       "tabbable": null,
       "tooltip": null,
       "value": " 22.9M/22.9M [00:00&lt;00:00, 36.5MB/s]"
      }
     }
    },
    "version_major": 2,
    "version_minor": 0
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
