{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e6d59019",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2026-03-03T19:10:51.006370Z",
     "iopub.status.busy": "2026-03-03T19:10:51.005752Z",
     "iopub.status.idle": "2026-03-03T19:10:57.448597Z",
     "shell.execute_reply": "2026-03-03T19:10:57.447947Z"
    },
    "papermill": {
     "duration": 6.448102,
     "end_time": "2026-03-03T19:10:57.450292",
     "exception": false,
     "start_time": "2026-03-03T19:10:51.002190",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Importing Libraries\n",
    "\n",
    "import os\n",
    "import random\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import librosa\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from sklearn.metrics import f1_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "0ee9b863",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-03T19:10:57.455077Z",
     "iopub.status.busy": "2026-03-03T19:10:57.454761Z",
     "iopub.status.idle": "2026-03-03T19:10:57.712361Z",
     "shell.execute_reply": "2026-03-03T19:10:57.711552Z"
    },
    "papermill": {
     "duration": 0.261647,
     "end_time": "2026-03-03T19:10:57.713900",
     "exception": false,
     "start_time": "2026-03-03T19:10:57.452253",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Executing on: cuda\n"
     ]
    }
   ],
   "source": [
    "# Configuration\n",
    "\n",
    "DATA_SEED = 67\n",
    "TRAINING_SEED = 1234\n",
    "\n",
    "# Audio\n",
    "SR = 22050\n",
    "DURATION = 5.0\n",
    "TOP_DB = 25\n",
    "N_MELS = 128\n",
    "N_FFT = 2048\n",
    "HOP_LENGTH = 512\n",
    "\n",
    "# Training\n",
    "BATCH_SIZE = 32\n",
    "EPOCHS = 30\n",
    "LEARNING_RATE = 1e-3\n",
    "\n",
    "DATA_ROOT = '/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup'\n",
    "GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']\n",
    "STEMS = {'drums': 'drums.wav', 'vocals': 'vocals.wav', 'bass': 'bass.wav', 'other': 'other.wav'}\n",
    "STEM_KEYS = ['drums', 'vocals', 'bass', 'other']\n",
    "\n",
    "random.seed(DATA_SEED)\n",
    "np.random.seed(DATA_SEED)\n",
    "torch.manual_seed(TRAINING_SEED)\n",
    "torch.cuda.manual_seed(TRAINING_SEED)\n",
    "\n",
    "DEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"Executing on: {DEVICE}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "91942b88",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-03T19:10:57.719040Z",
     "iopub.status.busy": "2026-03-03T19:10:57.718490Z",
     "iopub.status.idle": "2026-03-03T19:11:02.810170Z",
     "shell.execute_reply": "2026-03-03T19:11:02.809611Z"
    },
    "papermill": {
     "duration": 5.095999,
     "end_time": "2026-03-03T19:11:02.811820",
     "exception": false,
     "start_time": "2026-03-03T19:10:57.715821",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Train Test Split \n",
    "\n",
    "def build_dataset(root_dir, val_split=0.17, seed=42):\n",
    "    train_dict = {g: {k: [] for k in STEM_KEYS} for g in GENRES}\n",
    "    val_dict = {g: {k: [] for k in STEM_KEYS} for g in GENRES}\n",
    "    \n",
    "    rng = random.Random(seed)\n",
    "    base_path = os.path.join(root_dir, 'genres_stems')\n",
    "\n",
    "    for genre in GENRES:\n",
    "        genre_path = os.path.join(base_path, genre)\n",
    "        if not os.path.isdir(genre_path): continue\n",
    "\n",
    "        valid_songs = []\n",
    "        for song_folder in sorted(os.listdir(genre_path)):\n",
    "            song_path = os.path.join(genre_path, song_folder)\n",
    "            if not os.path.isdir(song_path): continue\n",
    "\n",
    "            song_stems = {}\n",
    "            is_valid = True\n",
    "            for key in STEM_KEYS:\n",
    "                stem_path = os.path.join(song_path, f\"{key}.wav\")\n",
    "                if not os.path.isfile(stem_path) or os.path.getsize(stem_path) < 4096:\n",
    "                    is_valid = False\n",
    "                    break\n",
    "                song_stems[key] = stem_path\n",
    "            \n",
    "            if is_valid: valid_songs.append(song_stems)\n",
    "\n",
    "        rng.shuffle(valid_songs)\n",
    "        split_idx = int(len(valid_songs) * (1 - val_split))\n",
    "        \n",
    "        for song in valid_songs[:split_idx]:\n",
    "            for key in STEM_KEYS:\n",
    "                train_dict[genre][key].append(song[key])\n",
    "        for song in valid_songs[split_idx:]:\n",
    "            for key in STEM_KEYS:\n",
    "                val_dict[genre][key].append(song[key])\n",
    "                \n",
    "    return train_dict, val_dict\n",
    "\n",
    "train_dict, val_dict = build_dataset(DATA_ROOT)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "cd6d98b2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-03T19:11:02.816854Z",
     "iopub.status.busy": "2026-03-03T19:11:02.816636Z",
     "iopub.status.idle": "2026-03-03T19:11:02.822837Z",
     "shell.execute_reply": "2026-03-03T19:11:02.822303Z"
    },
    "papermill": {
     "duration": 0.010233,
     "end_time": "2026-03-03T19:11:02.824107",
     "exception": false,
     "start_time": "2026-03-03T19:11:02.813874",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Detecting Silence \n",
    "\n",
    "def find_long_silences(dataset_dict, sr=SR, threshold_sec=DURATION, top_db=TOP_DB):\n",
    "    records = []\n",
    "    all_tasks = [(g, s, p) for g in dataset_dict for s, paths in dataset_dict[g].items() for p in paths]\n",
    "    \n",
    "    for genre, stem, path in all_tasks:\n",
    "        y, _ = librosa.load(path, sr=sr)\n",
    "        intervals = librosa.effects.split(y, top_db=top_db)\n",
    "        \n",
    "        total_silence = 0\n",
    "        if len(intervals) == 0:\n",
    "            total_silence = librosa.get_duration(y=y, sr=sr)\n",
    "        else:\n",
    "            intervals_sec = intervals / sr\n",
    "            total_silence += intervals_sec[0][0] # Start\n",
    "            for i in range(len(intervals_sec)-1):\n",
    "                total_silence += (intervals_sec[i+1][0] - intervals_sec[i][1]) # Mid\n",
    "            total_silence += (librosa.get_duration(y=y, sr=sr) - intervals_sec[-1][1]) # End\n",
    "\n",
    "        if total_silence >= threshold_sec:\n",
    "            records.append({\"Genre\": genre, \"Stem\": stem, \"Total_Silence_Sec\": total_silence, \"File_Path\": path})\n",
    "            \n",
    "    return pd.DataFrame(records)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "bc472383",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-03T19:11:02.828419Z",
     "iopub.status.busy": "2026-03-03T19:11:02.828194Z",
     "iopub.status.idle": "2026-03-03T19:11:02.833449Z",
     "shell.execute_reply": "2026-03-03T19:11:02.832800Z"
    },
    "papermill": {
     "duration": 0.009018,
     "end_time": "2026-03-03T19:11:02.834868",
     "exception": false,
     "start_time": "2026-03-03T19:11:02.825850",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "class ResBlock(nn.Module):\n",
    "    def __init__(self, in_channels, out_channels):\n",
    "        super().__init__()\n",
    "        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)\n",
    "        self.bn1 = nn.BatchNorm2d(out_channels)\n",
    "        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)\n",
    "        self.bn2 = nn.BatchNorm2d(out_channels)\n",
    "        \n",
    "        self.shortcut = nn.Sequential()\n",
    "        if in_channels != out_channels:\n",
    "            self.shortcut = nn.Sequential(\n",
    "                nn.Conv2d(in_channels, out_channels, kernel_size=1),\n",
    "                nn.BatchNorm2d(out_channels)\n",
    "            )\n",
    "\n",
    "    def forward(self, x):\n",
    "        residual = self.shortcut(x)\n",
    "        out = torch.relu(self.bn1(self.conv1(x)))\n",
    "        out = self.bn2(self.conv2(out))\n",
    "        out += residual\n",
    "        return torch.relu(out)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "2fc4dcc2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-03T19:11:02.839407Z",
     "iopub.status.busy": "2026-03-03T19:11:02.838851Z",
     "iopub.status.idle": "2026-03-03T19:11:02.843905Z",
     "shell.execute_reply": "2026-03-03T19:11:02.843360Z"
    },
    "papermill": {
     "duration": 0.008737,
     "end_time": "2026-03-03T19:11:02.845288",
     "exception": false,
     "start_time": "2026-03-03T19:11:02.836551",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Model\n",
    "\n",
    "class MusicResNet(nn.Module):\n",
    "    def __init__(self, num_classes=10):\n",
    "        super().__init__()\n",
    "        self.stage1 = ResBlock(1, 32)\n",
    "        self.stage2 = ResBlock(32, 64)\n",
    "        self.stage3 = ResBlock(64, 128)\n",
    "        self.stage4 = ResBlock(128, 256)\n",
    "        self.pool = nn.MaxPool2d(2)\n",
    "        \n",
    "        self.classifier = nn.Sequential(\n",
    "            nn.AdaptiveAvgPool2d(1),\n",
    "            nn.Flatten(),\n",
    "            nn.Dropout(0.4),\n",
    "            nn.Linear(256, 128),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(128, num_classes)\n",
    "        )\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.pool(self.stage1(x))\n",
    "        x = self.pool(self.stage2(x))\n",
    "        x = self.pool(self.stage3(x))\n",
    "        x = self.pool(self.stage4(x))\n",
    "        return self.classifier(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "8ca10d2b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-03T19:11:02.849710Z",
     "iopub.status.busy": "2026-03-03T19:11:02.849505Z",
     "iopub.status.idle": "2026-03-03T19:11:02.856446Z",
     "shell.execute_reply": "2026-03-03T19:11:02.855769Z"
    },
    "papermill": {
     "duration": 0.010727,
     "end_time": "2026-03-03T19:11:02.857789",
     "exception": false,
     "start_time": "2026-03-03T19:11:02.847062",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Dataset \n",
    "\n",
    "class AudioDataset(Dataset):\n",
    "    def __init__(self, data_dict, samples_per_genre=100, augment=True):\n",
    "        self.data_dict = data_dict\n",
    "        self.samples_per_genre = samples_per_genre\n",
    "        self.augment = augment\n",
    "        self.samples = [(idx, g) for idx, g in enumerate(GENRES) for _ in range(samples_per_genre)]\n",
    "\n",
    "    def __len__(self): return len(self.samples)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        genre_idx, genre_name = self.samples[idx]\n",
    "        mix = np.zeros(int(SR * DURATION))\n",
    "        \n",
    "        for k in STEM_KEYS:\n",
    "            path = random.choice(self.data_dict[genre_name][k])\n",
    "            y, _ = librosa.load(path, sr=SR, duration=DURATION)\n",
    "            if len(y) < len(mix): y = np.pad(y, (0, len(mix) - len(y)))\n",
    "            mix += y * random.uniform(0.6, 1.3) if self.augment else y\n",
    "\n",
    "        mel = librosa.feature.melspectrogram(y=mix, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)\n",
    "        mel_db = librosa.power_to_db(mel, ref=np.max)\n",
    "        \n",
    "        mel_db = (mel_db - (-40.0)) / 40.0 \n",
    "        tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)\n",
    "        \n",
    "        if self.augment and random.random() < 0.3: \n",
    "            tensor[:, random.randint(0, 100):random.randint(100, 128), :] = 0\n",
    "            \n",
    "        return tensor, genre_idx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "7c42d5ba",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-03T19:11:02.861928Z",
     "iopub.status.busy": "2026-03-03T19:11:02.861699Z",
     "iopub.status.idle": "2026-03-03T22:13:28.520978Z",
     "shell.execute_reply": "2026-03-03T22:13:28.520302Z"
    },
    "papermill": {
     "duration": 10945.671563,
     "end_time": "2026-03-03T22:13:28.530959",
     "exception": false,
     "start_time": "2026-03-03T19:11:02.859396",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 01 | Loss: 1.5243 | Macro F1: 0.5216\n",
      "Epoch 02 | Loss: 1.1495 | Macro F1: 0.4094\n",
      "Epoch 03 | Loss: 0.9493 | Macro F1: 0.4450\n",
      "Epoch 04 | Loss: 0.8335 | Macro F1: 0.5503\n",
      "Epoch 05 | Loss: 0.7323 | Macro F1: 0.6035\n",
      "Epoch 06 | Loss: 0.6635 | Macro F1: 0.6725\n",
      "Epoch 07 | Loss: 0.6026 | Macro F1: 0.5996\n",
      "Epoch 08 | Loss: 0.5270 | Macro F1: 0.7272\n",
      "Epoch 09 | Loss: 0.4973 | Macro F1: 0.6857\n",
      "Epoch 10 | Loss: 0.4505 | Macro F1: 0.7466\n",
      "Epoch 11 | Loss: 0.4266 | Macro F1: 0.7123\n",
      "Epoch 12 | Loss: 0.3860 | Macro F1: 0.6931\n",
      "Epoch 13 | Loss: 0.3902 | Macro F1: 0.7209\n",
      "Epoch 14 | Loss: 0.3406 | Macro F1: 0.7084\n",
      "Epoch 15 | Loss: 0.2768 | Macro F1: 0.7913\n",
      "Epoch 16 | Loss: 0.2461 | Macro F1: 0.8411\n",
      "Epoch 17 | Loss: 0.2374 | Macro F1: 0.8434\n",
      "Epoch 18 | Loss: 0.2276 | Macro F1: 0.8385\n",
      "Epoch 19 | Loss: 0.2114 | Macro F1: 0.8420\n",
      "Epoch 20 | Loss: 0.2147 | Macro F1: 0.8297\n",
      "Epoch 21 | Loss: 0.2120 | Macro F1: 0.7960\n",
      "Epoch 22 | Loss: 0.1669 | Macro F1: 0.8578\n",
      "Epoch 23 | Loss: 0.1545 | Macro F1: 0.8610\n",
      "Epoch 24 | Loss: 0.1441 | Macro F1: 0.8115\n",
      "Epoch 25 | Loss: 0.1450 | Macro F1: 0.8741\n",
      "Epoch 26 | Loss: 0.1537 | Macro F1: 0.8629\n",
      "Epoch 27 | Loss: 0.1533 | Macro F1: 0.8506\n",
      "Epoch 28 | Loss: 0.1533 | Macro F1: 0.8244\n",
      "Epoch 29 | Loss: 0.1360 | Macro F1: 0.8770\n",
      "Epoch 30 | Loss: 0.1313 | Macro F1: 0.8569\n"
     ]
    }
   ],
   "source": [
    "# Training Function\n",
    "\n",
    "train_dict, val_dict = build_dataset(DATA_ROOT)\n",
    "train_loader = DataLoader(AudioDataset(train_dict, samples_per_genre=500), batch_size=BATCH_SIZE, shuffle=True)\n",
    "val_loader = DataLoader(AudioDataset(val_dict, samples_per_genre=100, augment=False), batch_size=BATCH_SIZE)\n",
    "\n",
    "model = MusicResNet().to(DEVICE)\n",
    "optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)\n",
    "\n",
    "for epoch in range(EPOCHS):\n",
    "    model.train()\n",
    "    train_loss = 0\n",
    "    for x, y in train_loader:\n",
    "        x, y = x.to(DEVICE), y.to(DEVICE)\n",
    "        optimizer.zero_grad()\n",
    "        loss = criterion(model(x), y)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        train_loss += loss.item()\n",
    "    \n",
    "    model.eval()\n",
    "    preds, targets = [], []\n",
    "    with torch.no_grad():\n",
    "        for x, y in val_loader:\n",
    "            out = model(x.to(DEVICE))\n",
    "            preds.extend(torch.argmax(out, dim=1).cpu().numpy())\n",
    "            targets.extend(y.numpy())\n",
    "    \n",
    "    f1 = f1_score(targets, preds, average='macro')\n",
    "    scheduler.step(f1)\n",
    "    print(f\"Epoch {epoch+1:02d} | Loss: {train_loss/len(train_loader):.4f} | Macro F1: {f1:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "1ef9318c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-03T22:13:28.542527Z",
     "iopub.status.busy": "2026-03-03T22:13:28.542040Z",
     "iopub.status.idle": "2026-03-03T22:14:32.605815Z",
     "shell.execute_reply": "2026-03-03T22:14:32.605198Z"
    },
    "papermill": {
     "duration": 64.074557,
     "end_time": "2026-03-03T22:14:32.610475",
     "exception": false,
     "start_time": "2026-03-03T22:13:28.535918",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Success!\n"
     ]
    }
   ],
   "source": [
    "# Prediction and Submmission\n",
    "\n",
    "test_df = pd.read_csv('/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/test.csv')\n",
    "model.eval()\n",
    "test_preds = []\n",
    "\n",
    "with torch.no_grad():\n",
    "    for fname in test_df[\"filename\"]:\n",
    "        path = os.path.join(DATA_ROOT, fname)\n",
    "        y, _ = librosa.load(path, sr=SR, duration=DURATION)\n",
    "        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)\n",
    "        mel_db = (librosa.power_to_db(mel, ref=np.max) - (-40.0)) / 40.0\n",
    "        x = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)\n",
    "        test_preds.append(GENRES[torch.argmax(model(x), dim=1).item()])\n",
    "\n",
    "submission = pd.DataFrame({\n",
    "    \"id\": test_df[\"id\"], \n",
    "    \"genre\": test_preds\n",
    "})\n",
    "\n",
    "submission.to_csv(\"submission.csv\", index=False)\n",
    "\n",
    "print(\"Success!\")"
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
   "dockerImageVersionId": 31259,
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
   "duration": 11027.480356,
   "end_time": "2026-03-03T22:14:35.789862",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-03-03T19:10:48.309506",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
