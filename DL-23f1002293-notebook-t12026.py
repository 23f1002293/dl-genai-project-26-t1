{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "5c8684a8",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2026-02-28T11:18:30.475510Z",
     "iopub.status.busy": "2026-02-28T11:18:30.474634Z",
     "iopub.status.idle": "2026-02-28T11:18:39.665910Z",
     "shell.execute_reply": "2026-02-28T11:18:39.664779Z"
    },
    "papermill": {
     "duration": 9.198061,
     "end_time": "2026-02-28T11:18:39.668088",
     "exception": false,
     "start_time": "2026-02-28T11:18:30.470027",
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
    "from sklearn.metrics import f1_score\n",
    "from sklearn.utils.class_weight import compute_class_weight\n",
    "from sklearn.model_selection import StratifiedKFold"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b2a26e14",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-28T11:18:39.675417Z",
     "iopub.status.busy": "2026-02-28T11:18:39.674312Z",
     "iopub.status.idle": "2026-02-28T11:18:39.682980Z",
     "shell.execute_reply": "2026-02-28T11:18:39.681977Z"
    },
    "papermill": {
     "duration": 0.014363,
     "end_time": "2026-02-28T11:18:39.684943",
     "exception": false,
     "start_time": "2026-02-28T11:18:39.670580",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "cpu\n"
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
    "TOP_DB = 20\n",
    "N_MELS = 128\n",
    "N_FFT = 2048\n",
    "HOP_LENGTH = 512\n",
    "\n",
    "# Training\n",
    "BATCH_SIZE = 32\n",
    "EPOCHS = 20\n",
    "LEARNING_RATE = 0.001\n",
    "\n",
    "DATA_ROOT = '/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup'\n",
    "GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']\n",
    "STEMS = {'drums': 'drums.wav', 'vocals': 'vocals.wav', 'bass': 'bass.wav', 'other': 'other.wav'}\n",
    "STEM_KEYS = ['drums', 'vocals', 'bass', 'other']\n",
    "\n",
    "random.seed(DATA_SEED)\n",
    "np.random.seed(DATA_SEED)\n",
    "\n",
    "DEVICE = torch.device(\"cpu\")  \n",
    "print(DEVICE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "414ac587",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-28T11:18:39.691468Z",
     "iopub.status.busy": "2026-02-28T11:18:39.691163Z",
     "iopub.status.idle": "2026-02-28T11:18:45.854240Z",
     "shell.execute_reply": "2026-02-28T11:18:45.853263Z"
    },
    "papermill": {
     "duration": 6.168891,
     "end_time": "2026-02-28T11:18:45.856308",
     "exception": false,
     "start_time": "2026-02-28T11:18:39.687417",
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
   "id": "a2d52f39",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-28T11:18:45.863137Z",
     "iopub.status.busy": "2026-02-28T11:18:45.862606Z",
     "iopub.status.idle": "2026-02-28T11:18:45.870437Z",
     "shell.execute_reply": "2026-02-28T11:18:45.869735Z"
    },
    "papermill": {
     "duration": 0.013651,
     "end_time": "2026-02-28T11:18:45.872342",
     "exception": false,
     "start_time": "2026-02-28T11:18:45.858691",
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
   "id": "8208e048",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-28T11:18:45.878910Z",
     "iopub.status.busy": "2026-02-28T11:18:45.878411Z",
     "iopub.status.idle": "2026-02-28T11:18:45.887454Z",
     "shell.execute_reply": "2026-02-28T11:18:45.886742Z"
    },
    "papermill": {
     "duration": 0.014552,
     "end_time": "2026-02-28T11:18:45.889382",
     "exception": false,
     "start_time": "2026-02-28T11:18:45.874830",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Dataset \n",
    "\n",
    "class MashupDataset(Dataset):\n",
    "    def __init__(self, dataset_dict, genres, stem_keys, samples_per_genre=250):\n",
    "        self.dataset_dict = dataset_dict\n",
    "        self.genres = genres\n",
    "        self.stem_keys = stem_keys\n",
    "        self.samples = []\n",
    "        for idx, genre in enumerate(genres):\n",
    "            for _ in range(samples_per_genre):\n",
    "                self.samples.append((idx, genre))\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.samples)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        genre_idx, genre_name = self.samples[idx]\n",
    "        mix = np.zeros(int(SR * DURATION))\n",
    "\n",
    "        for key in self.stem_keys:\n",
    "            path = random.choice(self.dataset_dict[genre_name][key])\n",
    "            # Fast load\n",
    "            y, _ = librosa.load(path, sr=SR, duration=DURATION)\n",
    "            if len(y) < len(mix):\n",
    "                y = np.pad(y, (0, len(mix) - len(y)))\n",
    "            else:\n",
    "                y = y[:len(mix)]\n",
    "            \n",
    "            mix += y * random.uniform(0.7, 1.3)\n",
    "\n",
    "        mel = librosa.feature.melspectrogram(y=mix, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)\n",
    "        mel_db = librosa.power_to_db(mel, ref=np.max)\n",
    "        \n",
    "        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)\n",
    "\n",
    "        return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0), genre_idx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "88e12ee4",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-28T11:18:45.895863Z",
     "iopub.status.busy": "2026-02-28T11:18:45.895376Z",
     "iopub.status.idle": "2026-02-28T11:18:45.903011Z",
     "shell.execute_reply": "2026-02-28T11:18:45.902182Z"
    },
    "papermill": {
     "duration": 0.012851,
     "end_time": "2026-02-28T11:18:45.904691",
     "exception": false,
     "start_time": "2026-02-28T11:18:45.891840",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# CNN \n",
    "\n",
    "class GenreCNN(nn.Module):\n",
    "    def __init__(self, num_classes=10):\n",
    "        super().__init__()\n",
    "        self.features = nn.Sequential(\n",
    "            # Block 1\n",
    "            nn.Conv2d(1, 32, kernel_size=3, padding=1),\n",
    "            nn.BatchNorm2d(32),\n",
    "            nn.ReLU(),\n",
    "            nn.MaxPool2d(2),\n",
    "            \n",
    "            # Block 2\n",
    "            nn.Conv2d(32, 64, kernel_size=3, padding=1),\n",
    "            nn.BatchNorm2d(64),\n",
    "            nn.ReLU(),\n",
    "            nn.MaxPool2d(2),\n",
    "            \n",
    "            # Block 3\n",
    "            nn.Conv2d(64, 128, kernel_size=3, padding=1),\n",
    "            nn.BatchNorm2d(128),\n",
    "            nn.ReLU(),\n",
    "            nn.MaxPool2d(2),\n",
    "            \n",
    "            # Block 4\n",
    "            nn.Conv2d(128, 128, kernel_size=3, padding=1),\n",
    "            nn.BatchNorm2d(128),\n",
    "            nn.ReLU(),\n",
    "            nn.AdaptiveAvgPool2d((4, 4)) # Reduces spatial dims to fixed size\n",
    "        )\n",
    "\n",
    "        self.classifier = nn.Sequential(\n",
    "            nn.Flatten(),\n",
    "            nn.Linear(128 * 4 * 4, 256),\n",
    "            nn.ReLU(),\n",
    "            nn.Dropout(0.4), # Increased dropout for better generalization\n",
    "            nn.Linear(256, num_classes)\n",
    "        )\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.features(x)\n",
    "        return self.classifier(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "580b3427",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-28T11:18:45.911225Z",
     "iopub.status.busy": "2026-02-28T11:18:45.910670Z",
     "iopub.status.idle": "2026-02-28T11:18:45.916306Z",
     "shell.execute_reply": "2026-02-28T11:18:45.915196Z"
    },
    "papermill": {
     "duration": 0.010997,
     "end_time": "2026-02-28T11:18:45.918121",
     "exception": false,
     "start_time": "2026-02-28T11:18:45.907124",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Training Function\n",
    "\n",
    "def train_one_epoch(model, loader, criterion, optimizer):\n",
    "    model.train()\n",
    "    total_loss = 0\n",
    "    for inputs, labels in loader:\n",
    "        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)\n",
    "        optimizer.zero_grad()\n",
    "        loss = criterion(model(inputs), labels)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        total_loss += loss.item()\n",
    "    return total_loss / len(loader)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "74850b21",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-28T11:18:45.924587Z",
     "iopub.status.busy": "2026-02-28T11:18:45.924033Z",
     "iopub.status.idle": "2026-02-28T11:18:45.930418Z",
     "shell.execute_reply": "2026-02-28T11:18:45.929200Z"
    },
    "papermill": {
     "duration": 0.011904,
     "end_time": "2026-02-28T11:18:45.932364",
     "exception": false,
     "start_time": "2026-02-28T11:18:45.920460",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Evalutation Function\n",
    "\n",
    "def evaluate(model, loader):\n",
    "    model.eval()\n",
    "    preds, labels_list = [], []\n",
    "    with torch.no_grad():\n",
    "        for inputs, labels in loader:\n",
    "            output = model(inputs.to(DEVICE))\n",
    "            preds.extend(torch.argmax(output, dim=1).cpu().numpy())\n",
    "            labels_list.extend(labels.numpy())\n",
    "    return f1_score(labels_list, preds, average='macro')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "30eea540",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-28T11:18:45.939103Z",
     "iopub.status.busy": "2026-02-28T11:18:45.938435Z",
     "iopub.status.idle": "2026-02-28T14:23:14.790235Z",
     "shell.execute_reply": "2026-02-28T14:23:14.789130Z"
    },
    "papermill": {
     "duration": 11068.860661,
     "end_time": "2026-02-28T14:23:14.795429",
     "exception": false,
     "start_time": "2026-02-28T11:18:45.934768",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20 | Loss: 1.4880 | Val F1: 0.4707 | LR: 0.001000\n",
      "Epoch 2/20 | Loss: 1.0695 | Val F1: 0.5372 | LR: 0.001000\n",
      "Epoch 3/20 | Loss: 0.8436 | Val F1: 0.6822 | LR: 0.001000\n",
      "Epoch 4/20 | Loss: 0.7720 | Val F1: 0.5772 | LR: 0.001000\n",
      "Epoch 5/20 | Loss: 0.6374 | Val F1: 0.6133 | LR: 0.001000\n",
      "Epoch 6/20 | Loss: 0.5504 | Val F1: 0.5534 | LR: 0.001000\n",
      "Epoch 7/20 | Loss: 0.5062 | Val F1: 0.7494 | LR: 0.001000\n",
      "Epoch 8/20 | Loss: 0.4678 | Val F1: 0.6904 | LR: 0.001000\n",
      "Epoch 9/20 | Loss: 0.4107 | Val F1: 0.7218 | LR: 0.001000\n",
      "Epoch 10/20 | Loss: 0.3815 | Val F1: 0.6636 | LR: 0.001000\n",
      "Epoch 11/20 | Loss: 0.3503 | Val F1: 0.7066 | LR: 0.000500\n",
      "Epoch 12/20 | Loss: 0.2739 | Val F1: 0.7801 | LR: 0.000500\n",
      "Epoch 13/20 | Loss: 0.2162 | Val F1: 0.8259 | LR: 0.000500\n",
      "Epoch 14/20 | Loss: 0.1982 | Val F1: 0.7831 | LR: 0.000500\n",
      "Epoch 15/20 | Loss: 0.1952 | Val F1: 0.7312 | LR: 0.000500\n",
      "Epoch 16/20 | Loss: 0.1879 | Val F1: 0.7365 | LR: 0.000500\n",
      "Epoch 17/20 | Loss: 0.1847 | Val F1: 0.7616 | LR: 0.000250\n",
      "Epoch 18/20 | Loss: 0.1393 | Val F1: 0.7892 | LR: 0.000250\n",
      "Epoch 19/20 | Loss: 0.1407 | Val F1: 0.7883 | LR: 0.000250\n",
      "Epoch 20/20 | Loss: 0.1223 | Val F1: 0.8013 | LR: 0.000250\n"
     ]
    }
   ],
   "source": [
    "# DataLoaders\n",
    "\n",
    "train_loader = DataLoader(MashupDataset(train_dict, GENRES, STEM_KEYS, 400), batch_size=BATCH_SIZE, shuffle=True)\n",
    "val_loader = DataLoader(MashupDataset(val_dict, GENRES, STEM_KEYS, 100), batch_size=BATCH_SIZE)\n",
    "\n",
    "# Model, Loss, Optimizer\n",
    "\n",
    "model = GenreCNN().to(DEVICE)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)\n",
    "scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)\n",
    "\n",
    "# Training Loop\n",
    "\n",
    "for epoch in range(EPOCHS):\n",
    "    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)\n",
    "    val_f1 = evaluate(model, val_loader)\n",
    "    scheduler.step(val_f1)\n",
    "    print(f\"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "978c0899",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-28T14:23:14.804660Z",
     "iopub.status.busy": "2026-02-28T14:23:14.804180Z",
     "iopub.status.idle": "2026-02-28T14:27:46.839602Z",
     "shell.execute_reply": "2026-02-28T14:27:46.838791Z"
    },
    "papermill": {
     "duration": 272.047342,
     "end_time": "2026-02-28T14:27:46.846666",
     "exception": false,
     "start_time": "2026-02-28T14:23:14.799324",
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
    "model.eval()\n",
    "test_df = pd.read_csv('/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/test.csv') \n",
    "results = []\n",
    "\n",
    "with torch.no_grad():\n",
    "    for filename in test_df[\"filename\"]:\n",
    "        path = os.path.join(DATA_ROOT, filename)\n",
    "        y, _ = librosa.load(path, sr=SR, duration=DURATION)\n",
    "\n",
    "        mel = librosa.feature.melspectrogram(\n",
    "            y=y,\n",
    "            sr=SR,\n",
    "            n_fft=N_FFT,\n",
    "            hop_length=HOP_LENGTH,\n",
    "            n_mels=N_MELS\n",
    "        )\n",
    "\n",
    "        mel_db = librosa.power_to_db(mel, ref=np.max)\n",
    "        mel_db = (mel_db + 40.0) / 40.0\n",
    "\n",
    "        tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)\n",
    "        tensor = tensor.to(DEVICE)\n",
    "\n",
    "        output = model(tensor)\n",
    "        pred_idx = torch.argmax(output, dim=1).item()\n",
    "\n",
    "        results.append(GENRES[pred_idx])\n",
    "\n",
    "submission = pd.DataFrame({\n",
    "    \"id\": test_df[\"id\"],\n",
    "    \"genre\": results\n",
    "})\n",
    "\n",
    "submission.to_csv(\"submission.csv\", index=False)\n",
    "print(\"Success!\")"
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
   "dockerImageVersionId": 31259,
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
   "duration": 11363.622207,
   "end_time": "2026-02-28T14:27:50.385362",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-02-28T11:18:26.763155",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
