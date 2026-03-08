{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "535f6825",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2026-03-07T21:12:24.259639Z",
     "iopub.status.busy": "2026-03-07T21:12:24.259343Z",
     "iopub.status.idle": "2026-03-07T21:12:30.528432Z",
     "shell.execute_reply": "2026-03-07T21:12:30.527540Z"
    },
    "papermill": {
     "duration": 6.275083,
     "end_time": "2026-03-07T21:12:30.530378",
     "exception": false,
     "start_time": "2026-03-07T21:12:24.255295",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Importing Libraries \n",
    "\n",
    "import os\n",
    "import random\n",
    "import torch\n",
    "import torchaudio\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from sklearn.metrics import f1_score\n",
    "from tqdm import tqdm\n",
    "import soundfile as sf\n",
    "import subprocess\n",
    "import sys"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "a58ba8d0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-07T21:12:30.535878Z",
     "iopub.status.busy": "2026-03-07T21:12:30.535436Z",
     "iopub.status.idle": "2026-03-07T21:12:30.543947Z",
     "shell.execute_reply": "2026-03-07T21:12:30.543111Z"
    },
    "papermill": {
     "duration": 0.013466,
     "end_time": "2026-03-07T21:12:30.545952",
     "exception": false,
     "start_time": "2026-03-07T21:12:30.532486",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def get_dataset_mapping(root_dir):\n",
    "    genres_dir = os.path.join(root_dir, \"genres_stems\")\n",
    "    if not os.path.exists(genres_dir): return {}\n",
    "    genres = sorted([d for d in os.listdir(genres_dir) if os.path.isdir(os.path.join(genres_dir, d))])\n",
    "    dataset = {}\n",
    "    for genre in genres:\n",
    "        genre_path = os.path.join(genres_dir, genre)\n",
    "        songs = sorted([d for d in os.listdir(genre_path) if os.path.isdir(os.path.join(genre_path, d))])\n",
    "        song_mappings = {}\n",
    "        for song in songs:\n",
    "            song_path = os.path.join(genre_path, song)\n",
    "            stems = {\n",
    "                'bass': os.path.join(song_path, 'bass.wav'),\n",
    "                'drums': os.path.join(song_path, 'drums.wav'),\n",
    "                'other': os.path.join(song_path, 'other.wav'),\n",
    "                'vocals': os.path.join(song_path, 'vocals.wav')\n",
    "            }\n",
    "            stems = {k: v for k, v in stems.items() if os.path.exists(v)}\n",
    "            if len(stems) == 4:\n",
    "                song_mappings[song] = stems\n",
    "        dataset[genre] = song_mappings\n",
    "    return dataset\n",
    "\n",
    "def get_noise_files(root_dir):\n",
    "    noise_dir = os.path.join(root_dir, \"ESC-50-master\", \"audio\")\n",
    "    if not os.path.exists(noise_dir): return []\n",
    "    return [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.wav')]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c96b7981",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-07T21:12:30.550830Z",
     "iopub.status.busy": "2026-03-07T21:12:30.550538Z",
     "iopub.status.idle": "2026-03-07T21:12:30.555909Z",
     "shell.execute_reply": "2026-03-07T21:12:30.555083Z"
    },
    "papermill": {
     "duration": 0.009652,
     "end_time": "2026-03-07T21:12:30.557363",
     "exception": false,
     "start_time": "2026-03-07T21:12:30.547711",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    " # Configuration\n",
    "\n",
    "class CFG:\n",
    "    ROOT = \"/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup\"    \n",
    "    TEST_DIR = os.path.join(ROOT, \"/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/test.csv\")\n",
    "    OUTPUT_STEMS = \"/kaggle/working/separated_stems\" if os.path.exists(\"/kaggle\") else \"separated_stems\"\n",
    "    \n",
    "    SR = 16000\n",
    "    DURATION = 5\n",
    "    BATCH_SIZE = 32\n",
    "    EPOCHS = 40\n",
    "    LR = 3e-4\n",
    "    GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']\n",
    "    DEVICE = \"cuda\" if torch.cuda.is_available() else \"mps\" if torch.backends.mps.is_available() else \"cpu\"\n",
    "    SAMPLES_PER_EPOCH = 1200"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "33493fae",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-07T21:12:30.562282Z",
     "iopub.status.busy": "2026-03-07T21:12:30.561968Z",
     "iopub.status.idle": "2026-03-07T21:12:30.570119Z",
     "shell.execute_reply": "2026-03-07T21:12:30.568828Z"
    },
    "papermill": {
     "duration": 0.01252,
     "end_time": "2026-03-07T21:12:30.571657",
     "exception": false,
     "start_time": "2026-03-07T21:12:30.559137",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Preprocessing \n",
    "\n",
    "class InstrumentProcessor:\n",
    "    def __init__(self, mode, sr=16000):\n",
    "        self.sr = sr\n",
    "        self.mode = mode\n",
    "        if mode == 'bass':\n",
    "            self.transform = torchaudio.transforms.MelSpectrogram(\n",
    "                sample_rate=sr, n_fft=2048, win_length=2048, hop_length=512,\n",
    "                n_mels=32, f_min=20, f_max=250)\n",
    "        elif mode == 'vocals':\n",
    "            self.transform = torchaudio.transforms.MFCC(\n",
    "                sample_rate=sr, n_mfcc=40,\n",
    "                melkwargs={\"n_fft\": 1024, \"hop_length\": 512, \"n_mels\": 128})\n",
    "        elif mode == 'drums':\n",
    "            self.transform = torchaudio.transforms.MelSpectrogram(\n",
    "                sample_rate=sr, n_fft=1024, win_length=512, hop_length=256,\n",
    "                n_mels=128)\n",
    "        else:\n",
    "            self.transform = torchaudio.transforms.MelSpectrogram(\n",
    "                sample_rate=sr, n_fft=1024, hop_length=512, n_mels=128)\n",
    "        self.db_transform = torchaudio.transforms.AmplitudeToDB()\n",
    "\n",
    "    def __call__(self, wav):\n",
    "        feat = self.transform(wav)\n",
    "        if self.mode != 'vocals':\n",
    "            feat = self.db_transform(feat)\n",
    "        feat = (feat - feat.mean()) / (feat.std() + 1e-6)\n",
    "        return feat"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "c2d2c2ae",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-07T21:12:30.577971Z",
     "iopub.status.busy": "2026-03-07T21:12:30.577632Z",
     "iopub.status.idle": "2026-03-07T21:12:30.598088Z",
     "shell.execute_reply": "2026-03-07T21:12:30.597278Z"
    },
    "papermill": {
     "duration": 0.026175,
     "end_time": "2026-03-07T21:12:30.600092",
     "exception": false,
     "start_time": "2026-03-07T21:12:30.573917",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Dataset and Augmentation\n",
    "\n",
    "class FusionDataset(Dataset):\n",
    "    def __init__(self, mapping, duration=5, sr=16000, augment=True, size=500):\n",
    "        self.mapping = mapping\n",
    "        self.duration = duration\n",
    "        self.sr = sr\n",
    "        self.augment = augment\n",
    "        self.target_samples = int(duration * sr)\n",
    "        self.size = size\n",
    "        self.genre_list = CFG.GENRES\n",
    "        self.processors = {k: InstrumentProcessor(k) for k in ['bass', 'drums', 'vocals', 'other']}\n",
    "        self.audio_cache = {}\n",
    "        \n",
    "        all_stems = []\n",
    "        for g in mapping:\n",
    "            for s in mapping[g]:\n",
    "                for st in mapping[g][s]:\n",
    "                    all_stems.append((g, s, st, mapping[g][s][st]))\n",
    "        \n",
    "        for g, s, st, path in tqdm(all_stems, desc=\"Caching Stems\"):\n",
    "            wav, sr_orig = sf.read(path)\n",
    "            wav = torch.from_numpy(wav).float()\n",
    "            if len(wav.shape) > 1: wav = wav.mean(1)\n",
    "            if sr_orig != self.sr: wav = torchaudio.functional.resample(wav, sr_orig, self.sr)\n",
    "            if g not in self.audio_cache: self.audio_cache[g] = {}\n",
    "            if s not in self.audio_cache[g]: self.audio_cache[g][s] = {}\n",
    "            self.audio_cache[g][s][st] = wav\n",
    "\n",
    "    def __len__(self): return self.size\n",
    "    def __getitem__(self, idx):\n",
    "        genre = random.choice(self.genre_list)\n",
    "        label = self.genre_list.index(genre)\n",
    "        song_id = random.choice(list(self.mapping[genre].keys()))\n",
    "        stems_data = {}\n",
    "        for stem in ['drums', 'bass', 'vocals', 'other']:\n",
    "            y = self.audio_cache[genre][song_id][stem]\n",
    "            if len(y) > self.target_samples:\n",
    "                start = random.randint(0, len(y) - self.target_samples)\n",
    "                y = y[start : start + self.target_samples]\n",
    "            else:\n",
    "                y = torch.nn.functional.pad(y, (0, self.target_samples - len(y)))\n",
    "            if self.augment and random.random() < 0.4: y = y * random.uniform(0.7, 1.3)\n",
    "            spec = self.processors[stem](y.unsqueeze(0)).squeeze(0)\n",
    "            if self.augment:\n",
    "                if random.random() < 0.5: # Freq mask\n",
    "                    m = random.randint(5, 15)\n",
    "                    spec[random.randint(0, spec.shape[0]-m):][0:m, :] = 0\n",
    "                if random.random() < 0.5: # Time mask\n",
    "                    m = random.randint(10, 30)\n",
    "                    spec[:, random.randint(0, spec.shape[1]-m):][:, 0:m] = 0\n",
    "            stems_data[stem] = spec\n",
    "        return stems_data['drums'], stems_data['bass'], stems_data['vocals'], stems_data['other'], label\n",
    "\n",
    "class ResBlock(nn.Module):\n",
    "    def __init__(self, in_ch, out_ch):\n",
    "        super().__init__()\n",
    "        self.conv = nn.Sequential(\n",
    "            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),\n",
    "            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch))\n",
    "        self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1), nn.BatchNorm2d(out_ch)) if in_ch != out_ch else nn.Identity()\n",
    "        self.relu = nn.ReLU(); self.pool = nn.MaxPool2d(2)\n",
    "    def forward(self, x): return self.pool(self.relu(self.conv(x) + self.shortcut(x)))\n",
    "\n",
    "class StemEncoder(nn.Module):\n",
    "    def __init__(self, out_dim=256):\n",
    "        super().__init__()\n",
    "        self.net = nn.Sequential(ResBlock(1, 32), ResBlock(32, 64), ResBlock(64, 128), ResBlock(128, 256), nn.AdaptiveAvgPool2d(1), nn.Flatten())\n",
    "    def forward(self, x): \n",
    "        if len(x.shape) == 3: x = x.unsqueeze(1)\n",
    "        return self.net(x)\n",
    "\n",
    "class FusionNetwork(nn.Module):\n",
    "    def __init__(self, num_classes=10):\n",
    "        super().__init__()\n",
    "        self.drums_enc = StemEncoder(256); self.bass_enc = StemEncoder(256)\n",
    "        self.vocals_enc = StemEncoder(256); self.other_enc = StemEncoder(256)\n",
    "        self.attn = nn.Sequential(nn.Linear(256*4, 128), nn.ReLU(), nn.Linear(128, 4), nn.Softmax(dim=1))\n",
    "        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))\n",
    "    def forward(self, d, b, v, o):\n",
    "        d_f, b_f, v_f, o_f = self.drums_enc(d), self.bass_enc(b), self.vocals_enc(v), self.other_enc(o)\n",
    "        weights = self.attn(torch.cat([d_f, b_f, v_f, o_f], dim=1))\n",
    "        feats = torch.stack([d_f, b_f, v_f, o_f], dim=1)\n",
    "        return self.classifier(torch.sum(feats * weights.unsqueeze(-1), dim=1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "ba8972d3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-07T21:12:30.605073Z",
     "iopub.status.busy": "2026-03-07T21:12:30.604796Z",
     "iopub.status.idle": "2026-03-07T21:12:30.615852Z",
     "shell.execute_reply": "2026-03-07T21:12:30.615191Z"
    },
    "papermill": {
     "duration": 0.015251,
     "end_time": "2026-03-07T21:12:30.617282",
     "exception": false,
     "start_time": "2026-03-07T21:12:30.602031",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Pipeline\n",
    "\n",
    "def train_pipeline():\n",
    "    try: wandb.init(project=\"Mashup-Fusion\", config=CFG.__dict__)\n",
    "    except: pass\n",
    "    mapping = get_dataset_mapping(CFG.ROOT)\n",
    "    loader = DataLoader(FusionDataset(mapping, size=CFG.SAMPLES_PER_EPOCH), batch_size=CFG.BATCH_SIZE, shuffle=True)\n",
    "    model = FusionNetwork().to(CFG.DEVICE)\n",
    "    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR)\n",
    "    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)\n",
    "    for epoch in range(CFG.EPOCHS):\n",
    "        model.train()\n",
    "        for d, b, v, o, y in tqdm(loader, desc=f\"Epoch {epoch+1}\"):\n",
    "            d, b, v, o, y = [x.to(CFG.DEVICE) for x in [d, b, v, o, y]]\n",
    "            optimizer.zero_grad(); loss = criterion(model(d, b, v, o), y)\n",
    "            loss.backward(); optimizer.step()\n",
    "        torch.save(model.state_dict(), \"best_fusion_model.pth\")\n",
    "\n",
    "def inference_pipeline():\n",
    "    if not os.path.exists(CFG.OUTPUT_STEMS):\n",
    "        subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-U\", \"demucs\"], check=True)\n",
    "        subprocess.run([sys.executable, \"-m\", \"demucs\", \"-n\", \"htdemucs\", CFG.TEST_DIR, \"-o\", CFG.OUTPUT_STEMS], check=True)\n",
    "    model = FusionNetwork().to(CFG.DEVICE); model.load_state_dict(torch.load(\"best_fusion_model.pth\", map_location=CFG.DEVICE))\n",
    "    model.eval(); procs = {k: InstrumentProcessor(k) for k in ['bass', 'drums', 'vocals', 'other']}\n",
    "    root = os.path.join(CFG.OUTPUT_STEMS, \"htdemucs\")\n",
    "    folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]\n",
    "    results = []\n",
    "    with torch.no_grad():\n",
    "        for f in tqdm(folders):\n",
    "            stems = {}\n",
    "            for s in ['drums', 'bass', 'vocals', 'other']:\n",
    "                w, _ = torchaudio.load(os.path.join(root, f, f\"{s}.wav\"))\n",
    "                w = w.mean(0) if w.shape[0] > 1 else w.squeeze(0)\n",
    "                w = torch.nn.functional.pad(w[:CFG.DURATION*CFG.SR], (0, max(0, CFG.DURATION*CFG.SR - len(w))))\n",
    "                stems[s] = procs[s](w.unsqueeze(0)).to(CFG.DEVICE)\n",
    "            p = model(stems['drums'], stems['bass'], stems['vocals'], stems['other'])\n",
    "            results.append({\"filename\": f, \"genre\": CFG.GENRES[p.argmax(1).item()]})\n",
    "    pd.DataFrame(results).to_csv(\"submission.csv\", index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "04e8921b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-07T21:12:30.621606Z",
     "iopub.status.busy": "2026-03-07T21:12:30.621359Z",
     "iopub.status.idle": "2026-03-08T01:28:32.432304Z",
     "shell.execute_reply": "2026-03-08T01:28:32.425923Z"
    },
    "papermill": {
     "duration": 15361.81606,
     "end_time": "2026-03-08T01:28:32.434967",
     "exception": false,
     "start_time": "2026-03-07T21:12:30.618907",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Caching Stems: 100%|██████████| 4000/4000 [05:50<00:00, 11.42it/s]\n",
      "Epoch 1: 100%|██████████| 38/38 [06:01<00:00,  9.51s/it]\n",
      "Epoch 2: 100%|██████████| 38/38 [06:10<00:00,  9.75s/it]\n",
      "Epoch 3: 100%|██████████| 38/38 [05:55<00:00,  9.36s/it]\n",
      "Epoch 4: 100%|██████████| 38/38 [06:13<00:00,  9.83s/it]\n",
      "Epoch 5: 100%|██████████| 38/38 [05:58<00:00,  9.43s/it]\n",
      "Epoch 6: 100%|██████████| 38/38 [06:16<00:00,  9.91s/it]\n",
      "Epoch 7: 100%|██████████| 38/38 [06:10<00:00,  9.75s/it]\n",
      "Epoch 8: 100%|██████████| 38/38 [06:06<00:00,  9.66s/it]\n",
      "Epoch 9: 100%|██████████| 38/38 [06:20<00:00, 10.01s/it]\n",
      "Epoch 10: 100%|██████████| 38/38 [06:04<00:00,  9.58s/it]\n",
      "Epoch 11: 100%|██████████| 38/38 [06:40<00:00, 10.54s/it]\n",
      "Epoch 12: 100%|██████████| 38/38 [06:20<00:00, 10.01s/it]\n",
      "Epoch 13: 100%|██████████| 38/38 [06:02<00:00,  9.54s/it]\n",
      "Epoch 14: 100%|██████████| 38/38 [06:05<00:00,  9.62s/it]\n",
      "Epoch 15: 100%|██████████| 38/38 [06:15<00:00,  9.89s/it]\n",
      "Epoch 16: 100%|██████████| 38/38 [06:19<00:00,  9.99s/it]\n",
      "Epoch 17: 100%|██████████| 38/38 [06:25<00:00, 10.15s/it]\n",
      "Epoch 18: 100%|██████████| 38/38 [06:16<00:00,  9.90s/it]\n",
      "Epoch 19: 100%|██████████| 38/38 [06:12<00:00,  9.81s/it]\n",
      "Epoch 20: 100%|██████████| 38/38 [06:13<00:00,  9.82s/it]\n",
      "Epoch 21: 100%|██████████| 38/38 [06:10<00:00,  9.74s/it]\n",
      "Epoch 22: 100%|██████████| 38/38 [06:12<00:00,  9.80s/it]\n",
      "Epoch 23: 100%|██████████| 38/38 [06:04<00:00,  9.58s/it]\n",
      "Epoch 24: 100%|██████████| 38/38 [06:12<00:00,  9.79s/it]\n",
      "Epoch 25: 100%|██████████| 38/38 [06:16<00:00,  9.91s/it]\n",
      "Epoch 26: 100%|██████████| 38/38 [06:22<00:00, 10.08s/it]\n",
      "Epoch 27: 100%|██████████| 38/38 [06:29<00:00, 10.25s/it]\n",
      "Epoch 28: 100%|██████████| 38/38 [06:17<00:00,  9.94s/it]\n",
      "Epoch 29: 100%|██████████| 38/38 [06:18<00:00,  9.97s/it]\n",
      "Epoch 30: 100%|██████████| 38/38 [06:16<00:00,  9.92s/it]\n",
      "Epoch 31: 100%|██████████| 38/38 [06:15<00:00,  9.87s/it]\n",
      "Epoch 32: 100%|██████████| 38/38 [06:30<00:00, 10.27s/it]\n",
      "Epoch 33: 100%|██████████| 38/38 [06:29<00:00, 10.25s/it]\n",
      "Epoch 34: 100%|██████████| 38/38 [06:21<00:00, 10.03s/it]\n",
      "Epoch 35: 100%|██████████| 38/38 [06:23<00:00, 10.09s/it]\n",
      "Epoch 36: 100%|██████████| 38/38 [06:16<00:00,  9.92s/it]\n",
      "Epoch 37: 100%|██████████| 38/38 [06:09<00:00,  9.73s/it]\n",
      "Epoch 38: 100%|██████████| 38/38 [06:08<00:00,  9.69s/it]\n",
      "Epoch 39: 100%|██████████| 38/38 [06:12<00:00,  9.80s/it]\n",
      "Epoch 40: 100%|██████████| 38/38 [06:17<00:00,  9.95s/it]\n"
     ]
    }
   ],
   "source": [
    "MODES = {\"TRAIN\": True, \"INFERENCE\": os.path.exists(\"best_fusion_model.pth\")}\n",
    "\n",
    "if MODES[\"INFERENCE\"] and os.path.exists(CFG.TEST_DIR):\n",
    "    inference_pipeline()\n",
    "elif os.path.exists(os.path.join(CFG.ROOT, \"genres_stems\")):\n",
    "    train_pipeline()\n",
    "else:\n",
    "    print(\"Error\")"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "databundleVersionId": 15477148,
     "sourceId": 128431,
     "sourceType": "competition"
    }
   ],
   "isGpuEnabled": false,
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
   "duration": 15375.460576,
   "end_time": "2026-03-08T01:28:36.609819",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-03-07T21:12:21.149243",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
