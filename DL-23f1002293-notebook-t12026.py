{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "23c75e9c",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2026-02-27T11:49:46.505389Z",
     "iopub.status.busy": "2026-02-27T11:49:46.505161Z",
     "iopub.status.idle": "2026-02-27T11:49:57.115835Z",
     "shell.execute_reply": "2026-02-27T11:49:57.115210Z"
    },
    "papermill": {
     "duration": 10.615767,
     "end_time": "2026-02-27T11:49:57.117522",
     "exception": false,
     "start_time": "2026-02-27T11:49:46.501755",
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
   "id": "8f8caf86",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-27T11:49:57.122835Z",
     "iopub.status.busy": "2026-02-27T11:49:57.122481Z",
     "iopub.status.idle": "2026-02-27T11:49:57.128732Z",
     "shell.execute_reply": "2026-02-27T11:49:57.127953Z"
    },
    "papermill": {
     "duration": 0.010479,
     "end_time": "2026-02-27T11:49:57.130146",
     "exception": false,
     "start_time": "2026-02-27T11:49:57.119667",
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
   "id": "f2c2be2c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-27T11:49:57.135453Z",
     "iopub.status.busy": "2026-02-27T11:49:57.134806Z",
     "iopub.status.idle": "2026-02-27T11:49:57.142121Z",
     "shell.execute_reply": "2026-02-27T11:49:57.141571Z"
    },
    "papermill": {
     "duration": 0.011326,
     "end_time": "2026-02-27T11:49:57.143380",
     "exception": false,
     "start_time": "2026-02-27T11:49:57.132054",
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
    "            for key, filename in STEMS.items():\n",
    "                stem_path = os.path.join(song_path, filename)\n",
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
    "        def add_to_dict(target, songs):\n",
    "            for song in songs:\n",
    "                for key in STEM_KEYS:\n",
    "                    target[genre][key].append(song[key])\n",
    "\n",
    "        add_to_dict(train_dict, valid_songs[:split_idx])\n",
    "        add_to_dict(val_dict, valid_songs[split_idx:])\n",
    "        print(f\"Genre: {genre:10} | Train: {len(valid_songs[:split_idx]):3} | Val: {len(valid_songs[split_idx:]):3}\")\n",
    "\n",
    "    return train_dict, val_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "976f7824",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-27T11:49:57.148169Z",
     "iopub.status.busy": "2026-02-27T11:49:57.147955Z",
     "iopub.status.idle": "2026-02-27T11:49:57.153622Z",
     "shell.execute_reply": "2026-02-27T11:49:57.153054Z"
    },
    "papermill": {
     "duration": 0.009453,
     "end_time": "2026-02-27T11:49:57.154884",
     "exception": false,
     "start_time": "2026-02-27T11:49:57.145431",
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
   "id": "8b4f67fb",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-27T11:49:57.159450Z",
     "iopub.status.busy": "2026-02-27T11:49:57.159205Z",
     "iopub.status.idle": "2026-02-27T11:49:57.165500Z",
     "shell.execute_reply": "2026-02-27T11:49:57.164927Z"
    },
    "papermill": {
     "duration": 0.010192,
     "end_time": "2026-02-27T11:49:57.166899",
     "exception": false,
     "start_time": "2026-02-27T11:49:57.156707",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Dataset\n",
    "\n",
    "class MashupDataset(Dataset):\n",
    "    def __init__(self, dataset_dict, genres, stem_keys, samples_per_genre=200):\n",
    "        self.dataset_dict = dataset_dict\n",
    "        self.genres = genres\n",
    "        self.stem_keys = stem_keys\n",
    "        self.samples = []\n",
    "\n",
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
    "            y, _ = librosa.load(path, sr=SR, duration=DURATION)\n",
    "\n",
    "            if len(y) < len(mix):\n",
    "                y = np.pad(y, (0, len(mix) - len(y)))\n",
    "\n",
    "            mix += y * random.uniform(0.8, 1.2)\n",
    "\n",
    "        mel = librosa.feature.melspectrogram(\n",
    "            y=mix,\n",
    "            sr=SR,\n",
    "            n_fft=N_FFT,\n",
    "            hop_length=HOP_LENGTH,\n",
    "            n_mels=N_MELS\n",
    "        )\n",
    "\n",
    "        mel_db = librosa.power_to_db(mel, ref=np.max)\n",
    "        mel_db = (mel_db + 40.0) / 40.0\n",
    "\n",
    "        tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)\n",
    "\n",
    "        return tensor, genre_idx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "4bbf1378",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-27T11:49:57.171795Z",
     "iopub.status.busy": "2026-02-27T11:49:57.171360Z",
     "iopub.status.idle": "2026-02-27T11:49:57.176865Z",
     "shell.execute_reply": "2026-02-27T11:49:57.176188Z"
    },
    "papermill": {
     "duration": 0.009305,
     "end_time": "2026-02-27T11:49:57.178121",
     "exception": false,
     "start_time": "2026-02-27T11:49:57.168816",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# CNN \n",
    "\n",
    "class GenreCNN(nn.Module):\n",
    "    def __init__(self, num_classes=len(GENRES)):\n",
    "        super().__init__()\n",
    "\n",
    "        self.features = nn.Sequential(\n",
    "            nn.Conv2d(1, 16, 3, padding=1),\n",
    "            nn.BatchNorm2d(16),\n",
    "            nn.ReLU(),\n",
    "            nn.MaxPool2d(2),\n",
    "\n",
    "            nn.Conv2d(16, 32, 3, padding=1),\n",
    "            nn.BatchNorm2d(32),\n",
    "            nn.ReLU(),\n",
    "            nn.MaxPool2d(2),\n",
    "\n",
    "            nn.Conv2d(32, 64, 3, padding=1),\n",
    "            nn.BatchNorm2d(64),\n",
    "            nn.ReLU(),\n",
    "            nn.MaxPool2d(2),\n",
    "        )\n",
    "\n",
    "        self.classifier = nn.Sequential(\n",
    "            nn.Flatten(),\n",
    "            nn.Linear(64 * 16 * 27, 128),\n",
    "            nn.ReLU(),\n",
    "            nn.Dropout(0.3),\n",
    "            nn.Linear(128, num_classes)\n",
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
   "id": "91c90ee4",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-27T11:49:57.182937Z",
     "iopub.status.busy": "2026-02-27T11:49:57.182564Z",
     "iopub.status.idle": "2026-02-27T11:49:57.186597Z",
     "shell.execute_reply": "2026-02-27T11:49:57.186016Z"
    },
    "papermill": {
     "duration": 0.007886,
     "end_time": "2026-02-27T11:49:57.187871",
     "exception": false,
     "start_time": "2026-02-27T11:49:57.179985",
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
    "\n",
    "    for inputs, labels in loader:\n",
    "        inputs = inputs.to(DEVICE)\n",
    "        labels = labels.to(DEVICE)\n",
    "\n",
    "        optimizer.zero_grad()\n",
    "        outputs = model(inputs)\n",
    "        loss = criterion(outputs, labels)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "        total_loss += loss.item()\n",
    "\n",
    "    return total_loss / len(loader)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "a3fb50ea",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-27T11:49:57.192644Z",
     "iopub.status.busy": "2026-02-27T11:49:57.192195Z",
     "iopub.status.idle": "2026-02-27T11:49:57.196497Z",
     "shell.execute_reply": "2026-02-27T11:49:57.195824Z"
    },
    "papermill": {
     "duration": 0.008177,
     "end_time": "2026-02-27T11:49:57.197895",
     "exception": false,
     "start_time": "2026-02-27T11:49:57.189718",
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
    "    all_preds, all_labels = [], []\n",
    "\n",
    "    with torch.no_grad():\n",
    "        for inputs, labels in loader:\n",
    "            inputs = inputs.to(DEVICE)\n",
    "            outputs = model(inputs)\n",
    "            preds = torch.argmax(outputs, dim=1)\n",
    "\n",
    "            all_preds.extend(preds.cpu().numpy())\n",
    "            all_labels.extend(labels.numpy())\n",
    "\n",
    "    return f1_score(all_labels, all_preds, average='macro')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "e323efce",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-27T11:49:57.202829Z",
     "iopub.status.busy": "2026-02-27T11:49:57.202311Z",
     "iopub.status.idle": "2026-02-27T11:50:05.872361Z",
     "shell.execute_reply": "2026-02-27T11:50:05.871657Z"
    },
    "papermill": {
     "duration": 8.674113,
     "end_time": "2026-02-27T11:50:05.873922",
     "exception": false,
     "start_time": "2026-02-27T11:49:57.199809",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Genre: blues      | Train:  83 | Val:  17\n",
      "Genre: classical  | Train:  83 | Val:  17\n",
      "Genre: country    | Train:  83 | Val:  17\n",
      "Genre: disco      | Train:  83 | Val:  17\n",
      "Genre: hiphop     | Train:  83 | Val:  17\n",
      "Genre: jazz       | Train:  83 | Val:  17\n",
      "Genre: metal      | Train:  83 | Val:  17\n",
      "Genre: pop        | Train:  83 | Val:  17\n",
      "Genre: reggae     | Train:  83 | Val:  17\n",
      "Genre: rock       | Train:  83 | Val:  17\n"
     ]
    }
   ],
   "source": [
    "# Build data\n",
    "\n",
    "train_dict, val_dict = build_dataset(DATA_ROOT)\n",
    "\n",
    "train_loader = DataLoader(\n",
    "    MashupDataset(train_dict, GENRES, STEM_KEYS),\n",
    "    batch_size=BATCH_SIZE,\n",
    "    shuffle=True\n",
    ")\n",
    "\n",
    "val_loader = DataLoader(\n",
    "    MashupDataset(val_dict, GENRES, STEM_KEYS, samples_per_genre=50),\n",
    "    batch_size=BATCH_SIZE\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f93ac563",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-27T11:50:05.879750Z",
     "iopub.status.busy": "2026-02-27T11:50:05.879506Z",
     "iopub.status.idle": "2026-02-27T12:53:12.423169Z",
     "shell.execute_reply": "2026-02-27T12:53:12.422294Z"
    },
    "papermill": {
     "duration": 3786.551672,
     "end_time": "2026-02-27T12:53:12.428031",
     "exception": false,
     "start_time": "2026-02-27T11:50:05.876359",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20 | Loss: 3.0334 | Val Macro F1: 0.1773\n",
      "Epoch 2/20 | Loss: 2.1012 | Val Macro F1: 0.1692\n",
      "Epoch 3/20 | Loss: 1.9666 | Val Macro F1: 0.1327\n",
      "Epoch 4/20 | Loss: 1.8427 | Val Macro F1: 0.2494\n",
      "Epoch 5/20 | Loss: 1.7051 | Val Macro F1: 0.3684\n",
      "Epoch 6/20 | Loss: 1.6791 | Val Macro F1: 0.2992\n",
      "Epoch 7/20 | Loss: 1.6491 | Val Macro F1: 0.3905\n",
      "Epoch 8/20 | Loss: 1.5303 | Val Macro F1: 0.4252\n",
      "Epoch 9/20 | Loss: 1.5149 | Val Macro F1: 0.3112\n",
      "Epoch 10/20 | Loss: 1.4939 | Val Macro F1: 0.4314\n",
      "Epoch 11/20 | Loss: 1.4650 | Val Macro F1: 0.3868\n",
      "Epoch 12/20 | Loss: 1.4421 | Val Macro F1: 0.4768\n",
      "Epoch 13/20 | Loss: 1.4700 | Val Macro F1: 0.4789\n",
      "Epoch 14/20 | Loss: 1.4413 | Val Macro F1: 0.4363\n",
      "Epoch 15/20 | Loss: 1.3948 | Val Macro F1: 0.4480\n",
      "Epoch 16/20 | Loss: 1.4031 | Val Macro F1: 0.4773\n",
      "Epoch 17/20 | Loss: 1.3748 | Val Macro F1: 0.4736\n",
      "Epoch 18/20 | Loss: 1.3297 | Val Macro F1: 0.4785\n",
      "Epoch 19/20 | Loss: 1.3298 | Val Macro F1: 0.5104\n",
      "Epoch 20/20 | Loss: 1.3019 | Val Macro F1: 0.4142\n"
     ]
    }
   ],
   "source": [
    "# Model Training \n",
    "\n",
    "model = GenreCNN().to(DEVICE)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)\n",
    "\n",
    "for epoch in range(EPOCHS):\n",
    "    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)\n",
    "    val_f1 = evaluate(model, val_loader)\n",
    "\n",
    "    print(f\"Epoch {epoch+1}/{EPOCHS} | \"\n",
    "          f\"Loss: {train_loss:.4f} | \"\n",
    "          f\"Val Macro F1: {val_f1:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "d2099f88",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-27T12:53:12.435935Z",
     "iopub.status.busy": "2026-02-27T12:53:12.435537Z",
     "iopub.status.idle": "2026-02-27T12:56:55.418845Z",
     "shell.execute_reply": "2026-02-27T12:56:55.416241Z"
    },
    "papermill": {
     "duration": 222.990596,
     "end_time": "2026-02-27T12:56:55.422240",
     "exception": false,
     "start_time": "2026-02-27T12:53:12.431644",
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
   "duration": 4035.459812,
   "end_time": "2026-02-27T12:56:58.518950",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-02-27T11:49:43.059138",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
