{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c7682b51",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2026-03-05T11:12:30.106742Z",
     "iopub.status.busy": "2026-03-05T11:12:30.106506Z",
     "iopub.status.idle": "2026-03-05T11:12:36.683054Z",
     "shell.execute_reply": "2026-03-05T11:12:36.682455Z"
    },
    "papermill": {
     "duration": 6.581418,
     "end_time": "2026-03-05T11:12:36.684654",
     "exception": false,
     "start_time": "2026-03-05T11:12:30.103236",
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
   "id": "0bb6ac20",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-05T11:12:36.690774Z",
     "iopub.status.busy": "2026-03-05T11:12:36.690391Z",
     "iopub.status.idle": "2026-03-05T11:12:36.955111Z",
     "shell.execute_reply": "2026-03-05T11:12:36.954310Z"
    },
    "papermill": {
     "duration": 0.268607,
     "end_time": "2026-03-05T11:12:36.956458",
     "exception": false,
     "start_time": "2026-03-05T11:12:36.687851",
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
   "id": "b1c514a4",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-05T11:12:36.961340Z",
     "iopub.status.busy": "2026-03-05T11:12:36.960934Z",
     "iopub.status.idle": "2026-03-05T11:12:42.405443Z",
     "shell.execute_reply": "2026-03-05T11:12:42.404655Z"
    },
    "papermill": {
     "duration": 5.44879,
     "end_time": "2026-03-05T11:12:42.407109",
     "exception": false,
     "start_time": "2026-03-05T11:12:36.958319",
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
   "id": "5181e73b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-05T11:12:42.412477Z",
     "iopub.status.busy": "2026-03-05T11:12:42.411984Z",
     "iopub.status.idle": "2026-03-05T11:12:42.418178Z",
     "shell.execute_reply": "2026-03-05T11:12:42.417582Z"
    },
    "papermill": {
     "duration": 0.010455,
     "end_time": "2026-03-05T11:12:42.419620",
     "exception": false,
     "start_time": "2026-03-05T11:12:42.409165",
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
   "id": "46cfe9c8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-05T11:12:42.424283Z",
     "iopub.status.busy": "2026-03-05T11:12:42.423729Z",
     "iopub.status.idle": "2026-03-05T11:12:42.428923Z",
     "shell.execute_reply": "2026-03-05T11:12:42.428349Z"
    },
    "papermill": {
     "duration": 0.008913,
     "end_time": "2026-03-05T11:12:42.430193",
     "exception": false,
     "start_time": "2026-03-05T11:12:42.421280",
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
    "        self.shortcut = nn.Sequential()\n",
    "        if in_channels != out_channels:\n",
    "            self.shortcut = nn.Sequential(\n",
    "                nn.Conv2d(in_channels, out_channels, kernel_size=1),\n",
    "                nn.BatchNorm2d(out_channels)\n",
    "            )\n",
    "    def forward(self, x):\n",
    "        residual = self.shortcut(x)\n",
    "        out = torch.relu(self.bn1(self.conv1(x)))\n",
    "        out = self.bn2(self.conv2(out))\n",
    "        out += residual\n",
    "        return torch.relu(out)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b46f5381",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-05T11:12:42.434691Z",
     "iopub.status.busy": "2026-03-05T11:12:42.434242Z",
     "iopub.status.idle": "2026-03-05T11:12:42.440234Z",
     "shell.execute_reply": "2026-03-05T11:12:42.439598Z"
    },
    "papermill": {
     "duration": 0.009757,
     "end_time": "2026-03-05T11:12:42.441627",
     "exception": false,
     "start_time": "2026-03-05T11:12:42.431870",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Model\n",
    "\n",
    "class MusicCRNN(nn.Module):\n",
    "    def __init__(self, num_classes=10):\n",
    "        super().__init__()\n",
    "        self.stage1 = ResBlock(1, 32)\n",
    "        self.stage2 = ResBlock(32, 64)\n",
    "        self.stage3 = ResBlock(64, 128)\n",
    "        self.stage4 = ResBlock(128, 256)\n",
    "        self.pool = nn.MaxPool2d(2)\n",
    "        \n",
    "        self.gru = nn.GRU(input_size=2048, hidden_size=128, \n",
    "                          num_layers=2, batch_first=True, bidirectional=True)\n",
    "        \n",
    "        self.classifier = nn.Sequential(\n",
    "            nn.Dropout(0.4),\n",
    "            nn.Linear(128 * 2, 64), \n",
    "            nn.ReLU(),\n",
    "            nn.Linear(64, num_classes)\n",
    "        )\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.pool(self.stage1(x))\n",
    "        x = self.pool(self.stage2(x))\n",
    "        x = self.pool(self.stage3(x))\n",
    "        x = self.pool(self.stage4(x))\n",
    "        \n",
    "        b, c, h, w = x.size()\n",
    "        x = x.permute(0, 3, 1, 2).contiguous() \n",
    "        x = x.view(b, w, c * h)\n",
    "        \n",
    "        x, _ = self.gru(x)\n",
    "        x = torch.mean(x, dim=1) \n",
    "        return self.classifier(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "bc202c27",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-05T11:12:42.446322Z",
     "iopub.status.busy": "2026-03-05T11:12:42.445751Z",
     "iopub.status.idle": "2026-03-05T11:12:42.452757Z",
     "shell.execute_reply": "2026-03-05T11:12:42.452210Z"
    },
    "papermill": {
     "duration": 0.010752,
     "end_time": "2026-03-05T11:12:42.454069",
     "exception": false,
     "start_time": "2026-03-05T11:12:42.443317",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Dataset \n",
    "\n",
    "class AudioDataset(Dataset):\n",
    "    def __init__(self, data_dict, samples_per_genre=500, augment=True):\n",
    "        self.data_dict = data_dict\n",
    "        self.samples = [(idx, g) for idx, g in enumerate(GENRES) for _ in range(samples_per_genre)]\n",
    "        self.augment = augment\n",
    "\n",
    "    def __len__(self): return len(self.samples)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        genre_idx, genre_name = self.samples[idx]\n",
    "        mix = np.zeros(int(SR * DURATION))\n",
    "        for k in STEM_KEYS:\n",
    "            path = random.choice(self.data_dict[genre_name][k])\n",
    "            y, _ = librosa.load(path, sr=SR, duration=DURATION)\n",
    "            if len(y) < len(mix): y = np.pad(y, (0, len(mix) - len(y)))\n",
    "            mix += y * (random.uniform(0.6, 1.3) if self.augment else 1.0)\n",
    "\n",
    "        mel = librosa.feature.melspectrogram(y=mix, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)\n",
    "        mel_db = (librosa.power_to_db(mel, ref=np.max) - (-40.0)) / 40.0 \n",
    "        tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)\n",
    "        \n",
    "        if self.augment and random.random() < 0.3:\n",
    "            # Frequency masking\n",
    "            tensor[:, random.randint(0, 80):random.randint(80, 128), :] = 0\n",
    "            # Time masking\n",
    "            tensor[:, :, random.randint(0, 100):random.randint(100, 200)] = 0\n",
    "            \n",
    "        return tensor, genre_idx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "c795332d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-05T11:12:42.458394Z",
     "iopub.status.busy": "2026-03-05T11:12:42.458184Z",
     "iopub.status.idle": "2026-03-05T14:07:49.861996Z",
     "shell.execute_reply": "2026-03-05T14:07:49.861385Z"
    },
    "papermill": {
     "duration": 10507.412315,
     "end_time": "2026-03-05T14:07:49.868072",
     "exception": false,
     "start_time": "2026-03-05T11:12:42.455757",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 01 | Loss: 1.6529 | Macro F1: 0.5809\n",
      "Epoch 02 | Loss: 1.2044 | Macro F1: 0.4344\n",
      "Epoch 03 | Loss: 1.0373 | Macro F1: 0.6202\n",
      "Epoch 04 | Loss: 0.9734 | Macro F1: 0.6445\n",
      "Epoch 05 | Loss: 0.8596 | Macro F1: 0.6803\n",
      "Epoch 06 | Loss: 0.7890 | Macro F1: 0.6263\n",
      "Epoch 07 | Loss: 0.7155 | Macro F1: 0.6366\n",
      "Epoch 08 | Loss: 0.6917 | Macro F1: 0.7024\n",
      "Epoch 09 | Loss: 0.6556 | Macro F1: 0.7715\n",
      "Epoch 10 | Loss: 0.6118 | Macro F1: 0.7727\n",
      "Epoch 11 | Loss: 0.5597 | Macro F1: 0.6607\n",
      "Epoch 12 | Loss: 0.5549 | Macro F1: 0.5513\n",
      "Epoch 13 | Loss: 0.5651 | Macro F1: 0.6930\n",
      "Epoch 14 | Loss: 0.5055 | Macro F1: 0.7209\n",
      "Epoch 15 | Loss: 0.3919 | Macro F1: 0.6792\n",
      "Epoch 16 | Loss: 0.3735 | Macro F1: 0.8138\n",
      "Epoch 17 | Loss: 0.3596 | Macro F1: 0.8129\n",
      "Epoch 18 | Loss: 0.3536 | Macro F1: 0.8218\n",
      "Epoch 19 | Loss: 0.3041 | Macro F1: 0.7611\n",
      "Epoch 20 | Loss: 0.3076 | Macro F1: 0.7993\n",
      "Epoch 21 | Loss: 0.3153 | Macro F1: 0.8059\n",
      "Epoch 22 | Loss: 0.3057 | Macro F1: 0.7983\n",
      "Epoch 23 | Loss: 0.2346 | Macro F1: 0.8428\n",
      "Epoch 24 | Loss: 0.2407 | Macro F1: 0.8425\n",
      "Epoch 25 | Loss: 0.2430 | Macro F1: 0.8506\n",
      "Epoch 26 | Loss: 0.2092 | Macro F1: 0.8384\n",
      "Epoch 27 | Loss: 0.2178 | Macro F1: 0.8590\n",
      "Epoch 28 | Loss: 0.2326 | Macro F1: 0.8380\n",
      "Epoch 29 | Loss: 0.1932 | Macro F1: 0.8380\n",
      "Epoch 30 | Loss: 0.2013 | Macro F1: 0.8507\n"
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
    "model = MusicCRNN().to(DEVICE)\n",
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
   "id": "e4904238",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-05T14:07:49.878751Z",
     "iopub.status.busy": "2026-03-05T14:07:49.878297Z",
     "iopub.status.idle": "2026-03-05T14:09:04.113157Z",
     "shell.execute_reply": "2026-03-05T14:09:04.112557Z"
    },
    "papermill": {
     "duration": 74.242328,
     "end_time": "2026-03-05T14:09:04.114919",
     "exception": false,
     "start_time": "2026-03-05T14:07:49.872591",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted 500/3020\n",
      "Predicted 1000/3020\n",
      "Predicted 1500/3020\n",
      "Predicted 2000/3020\n",
      "Predicted 2500/3020\n",
      "Predicted 3000/3020\n",
      "Success!\n"
     ]
    }
   ],
   "source": [
    "# Prediction and Submmission\n",
    "\n",
    "model.eval()\n",
    "\n",
    "test_df = pd.read_csv(os.path.join(DATA_ROOT, '/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/test.csv'))\n",
    "test_preds = []\n",
    "\n",
    "with torch.no_grad():\n",
    "    for i, fname in enumerate(test_df[\"filename\"]):\n",
    "        path = os.path.join(DATA_ROOT, fname)\n",
    "        y, _ = librosa.load(path, sr=SR, duration=DURATION)\n",
    "        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)\n",
    "        mel_db = (librosa.power_to_db(mel, ref=np.max) - (-40.0)) / 40.0\n",
    "        x = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)\n",
    "        \n",
    "        test_preds.append(GENRES[torch.argmax(model(x), dim=1).item()])\n",
    "        if (i+1) % 500 == 0: print(f\"Predicted {i+1}/{len(test_df)}\")\n",
    "\n",
    "submission = pd.DataFrame({\"id\": test_df[\"id\"], \"genre\": test_preds})\n",
    "submission.to_csv(\"submission.csv\", index=False)\n",
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
   "duration": 10599.94692,
   "end_time": "2026-03-05T14:09:07.397055",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-03-05T11:12:27.450135",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
