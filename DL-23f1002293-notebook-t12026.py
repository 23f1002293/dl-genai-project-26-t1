{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d7659b37",
   "metadata": {
    "_cell_guid": "29e03169-88d3-4f63-a7da-f34ae0eb3db0",
    "_uuid": "22e946c8-8912-4165-86f7-6fca72dacd53",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2026-03-09T15:44:37.030925Z",
     "iopub.status.busy": "2026-03-09T15:44:37.029817Z",
     "iopub.status.idle": "2026-03-09T15:44:57.436577Z",
     "shell.execute_reply": "2026-03-09T15:44:57.435601Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 20.416397,
     "end_time": "2026-03-09T15:44:57.438740",
     "exception": false,
     "start_time": "2026-03-09T15:44:37.022343",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import os\n",
    "import random\n",
    "import glob\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "import torch.nn.functional as F\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from torchvision.models import resnet18\n",
    "import librosa\n",
    "from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay\n",
    "from tqdm import tqdm\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib\n",
    "\n",
    "matplotlib.use('Agg')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "790f79fa",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-09T15:44:57.445641Z",
     "iopub.status.busy": "2026-03-09T15:44:57.445148Z",
     "iopub.status.idle": "2026-03-09T15:44:57.450830Z",
     "shell.execute_reply": "2026-03-09T15:44:57.449992Z"
    },
    "papermill": {
     "duration": 0.011079,
     "end_time": "2026-03-09T15:44:57.452481",
     "exception": false,
     "start_time": "2026-03-09T15:44:57.441402",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Configuration\n",
    "\n",
    "GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']\n",
    "GENRE_TO_IDX = {genre: i for i, genre in enumerate(GENRES)}\n",
    "STEMS = ['drums.wav', 'vocals.wav', 'bass.wav', 'other.wav']\n",
    "SAMPLE_RATE = 22050\n",
    "DURATION = 30\n",
    "N_SAMPLES = SAMPLE_RATE * DURATION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "406df0be",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-09T15:44:57.458815Z",
     "iopub.status.busy": "2026-03-09T15:44:57.458476Z",
     "iopub.status.idle": "2026-03-09T15:44:57.467349Z",
     "shell.execute_reply": "2026-03-09T15:44:57.466381Z"
    },
    "papermill": {
     "duration": 0.014301,
     "end_time": "2026-03-09T15:44:57.469174",
     "exception": false,
     "start_time": "2026-03-09T15:44:57.454873",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Model\n",
    "\n",
    "class AudioAttention(nn.Module):\n",
    "    def __init__(self, in_channels):\n",
    "        super(AudioAttention, self).__init__()\n",
    "        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)\n",
    "        \n",
    "    def forward(self, x):\n",
    "        weights = torch.sigmoid(self.conv(x))\n",
    "        return x * weights\n",
    "\n",
    "class GenreClassifier(nn.Module):\n",
    "    def __init__(self, num_classes=10):\n",
    "        super(GenreClassifier, self).__init__()\n",
    "        self.backbone = resnet18(weights=None)\n",
    "        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)\n",
    "        self.attention = AudioAttention(512)\n",
    "        self.backbone.fc = nn.Sequential(\n",
    "            nn.Linear(512, 256),\n",
    "            nn.ReLU(),\n",
    "            nn.Dropout(0.3),\n",
    "            nn.Linear(256, num_classes)\n",
    "        )\n",
    "        \n",
    "    def forward(self, x):\n",
    "        x = self.backbone.conv1(x)\n",
    "        x = self.backbone.bn1(x)\n",
    "        x = self.backbone.relu(x)\n",
    "        x = self.backbone.maxpool(x)\n",
    "        x = self.backbone.layer1(x)\n",
    "        x = self.backbone.layer2(x)\n",
    "        x = self.backbone.layer3(x)\n",
    "        x = self.backbone.layer4(x)\n",
    "        x = self.attention(x)\n",
    "        x = self.backbone.avgpool(x)\n",
    "        x = torch.flatten(x, 1)\n",
    "        x = self.backbone.fc(x)\n",
    "        return x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "e5910e6b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-09T15:44:57.475297Z",
     "iopub.status.busy": "2026-03-09T15:44:57.474962Z",
     "iopub.status.idle": "2026-03-09T15:44:57.487988Z",
     "shell.execute_reply": "2026-03-09T15:44:57.486695Z"
    },
    "papermill": {
     "duration": 0.01883,
     "end_time": "2026-03-09T15:44:57.490346",
     "exception": false,
     "start_time": "2026-03-09T15:44:57.471516",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Dataset and Model\n",
    "\n",
    "class MessyMashupDataset(Dataset):\n",
    "    def __init__(self, data_root, esc50_root, n_items=1000):\n",
    "        self.data_root = data_root\n",
    "        self.esc50_root = esc50_root\n",
    "        self.n_items = n_items\n",
    "        \n",
    "        self.genre_stems = {genre: [] for genre in GENRES}\n",
    "        for genre in GENRES:\n",
    "            genre_path = os.path.join(data_root, 'genres_stems', genre)\n",
    "            if not os.path.exists(genre_path): continue\n",
    "            song_dirs = [d for d in os.listdir(genre_path) if os.path.isdir(os.path.join(genre_path, d))]\n",
    "            for song_id in song_dirs:\n",
    "                self.genre_stems[genre].append(os.path.join(genre_path, song_id))\n",
    "        \n",
    "        self.noise_files = glob.glob(os.path.join(esc50_root, 'audio', '*.wav'))\n",
    "        \n",
    "    def __len__(self):\n",
    "        return self.n_items\n",
    "\n",
    "    def load_audio(self, path):\n",
    "        try:\n",
    "            waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)\n",
    "        except Exception:\n",
    "            waveform = np.zeros(N_SAMPLES, dtype=np.float32)\n",
    "            \n",
    "        if len(waveform) < N_SAMPLES:\n",
    "            waveform = np.pad(waveform, (0, max(0, N_SAMPLES - len(waveform))))\n",
    "        else:\n",
    "            waveform = waveform[:N_SAMPLES]\n",
    "        return torch.from_numpy(waveform).unsqueeze(0)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        genre = random.choice(GENRES)\n",
    "        label = GENRE_TO_IDX[genre]\n",
    "        stems_waveforms = []\n",
    "        for stem_name in STEMS:\n",
    "            song_dir = random.choice(self.genre_stems[genre])\n",
    "            stem_path = os.path.join(song_dir, stem_name)\n",
    "            if not os.path.exists(stem_path) and stem_name == 'other.wav':\n",
    "                stem_path = os.path.join(song_dir, 'others.wav')\n",
    "            stems_waveforms.append(self.load_audio(stem_path))\n",
    "        \n",
    "        mixed_wave = torch.mean(torch.stack(stems_waveforms), dim=0)\n",
    "        \n",
    "        if self.noise_files:\n",
    "            noise_wave = self.load_audio(random.choice(self.noise_files))\n",
    "            mixed_wave = mixed_wave + (noise_wave * random.uniform(0.05, 0.2))\n",
    "        \n",
    "        mixed_np = mixed_wave.squeeze().numpy()\n",
    "        mel_spec = librosa.feature.melspectrogram(y=mixed_np, sr=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=128)\n",
    "        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)\n",
    "        return torch.from_numpy(mel_spec_db).unsqueeze(0), label\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "95a4b955",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-09T15:44:57.496721Z",
     "iopub.status.busy": "2026-03-09T15:44:57.496385Z",
     "iopub.status.idle": "2026-03-09T15:44:57.504736Z",
     "shell.execute_reply": "2026-03-09T15:44:57.503976Z"
    },
    "papermill": {
     "duration": 0.013723,
     "end_time": "2026-03-09T15:44:57.506461",
     "exception": false,
     "start_time": "2026-03-09T15:44:57.492738",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "class TestMashupDataset(Dataset):\n",
    "    def __init__(self, data_root, test_csv):\n",
    "        self.data_root = data_root\n",
    "        self.df = pd.read_csv(test_csv)\n",
    "        \n",
    "    def __len__(self):\n",
    "        return len(self.df)\n",
    "    \n",
    "    def load_audio(self, path):\n",
    "        try:\n",
    "            waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)\n",
    "        except:\n",
    "            waveform = np.zeros(N_SAMPLES, dtype=np.float32)\n",
    "        if len(waveform) < N_SAMPLES:\n",
    "            waveform = np.pad(waveform, (0, max(0, N_SAMPLES - len(waveform))))\n",
    "        else:\n",
    "            waveform = waveform[:N_SAMPLES]\n",
    "        return torch.from_numpy(waveform).unsqueeze(0)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        filename = self.df.iloc[idx]['filename']\n",
    "        song_id = self.df.iloc[idx]['id']\n",
    "        path = os.path.join(self.data_root, filename)\n",
    "        wave = self.load_audio(path)\n",
    "        mixed_np = wave.squeeze().numpy()\n",
    "        mel_spec = librosa.feature.melspectrogram(y=mixed_np, sr=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=128)\n",
    "        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)\n",
    "        return torch.from_numpy(mel_spec_db).unsqueeze(0), song_id\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "fbd42fad",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-09T15:44:57.512760Z",
     "iopub.status.busy": "2026-03-09T15:44:57.512420Z",
     "iopub.status.idle": "2026-03-09T15:44:57.519144Z",
     "shell.execute_reply": "2026-03-09T15:44:57.518328Z"
    },
    "papermill": {
     "duration": 0.012008,
     "end_time": "2026-03-09T15:44:57.520874",
     "exception": false,
     "start_time": "2026-03-09T15:44:57.508866",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def train_epoch(model, loader, optimizer, criterion, device):\n",
    "    model.train()\n",
    "    running_loss, all_preds, all_labels = 0.0, [], []\n",
    "    pbar = tqdm(loader, desc=\"Training\")\n",
    "    for inputs, labels in pbar:\n",
    "        inputs, labels = inputs.to(device), labels.to(device)\n",
    "        optimizer.zero_grad()\n",
    "        outputs = model(inputs)\n",
    "        loss = criterion(outputs, labels)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        running_loss += loss.item()\n",
    "        preds = torch.argmax(outputs, dim=1)\n",
    "        all_preds.extend(preds.cpu().numpy())\n",
    "        all_labels.extend(labels.cpu().numpy())\n",
    "        pbar.set_postfix(loss=loss.item())\n",
    "    return running_loss / len(loader), f1_score(all_labels, all_preds, average='macro')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "4cbce3af",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-09T15:44:57.527303Z",
     "iopub.status.busy": "2026-03-09T15:44:57.526934Z",
     "iopub.status.idle": "2026-03-09T15:44:57.533625Z",
     "shell.execute_reply": "2026-03-09T15:44:57.532797Z"
    },
    "papermill": {
     "duration": 0.012112,
     "end_time": "2026-03-09T15:44:57.535444",
     "exception": false,
     "start_time": "2026-03-09T15:44:57.523332",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def validate(model, loader, criterion, device):\n",
    "    model.eval()\n",
    "    running_loss, all_preds, all_labels = 0.0, [], []\n",
    "    with torch.no_grad():\n",
    "        for inputs, labels in tqdm(loader, desc=\"Validation\"):\n",
    "            inputs, labels = inputs.to(device), labels.to(device)\n",
    "            outputs = model(inputs)\n",
    "            loss = criterion(outputs, labels)\n",
    "            running_loss += loss.item()\n",
    "            preds = torch.argmax(outputs, dim=1)\n",
    "            all_preds.extend(preds.cpu().numpy())\n",
    "            all_labels.extend(labels.cpu().numpy())\n",
    "    return running_loss / len(loader), f1_score(all_labels, all_preds, average='macro'), all_labels, all_preds"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "4ce636e6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-09T15:44:57.541826Z",
     "iopub.status.busy": "2026-03-09T15:44:57.541182Z",
     "iopub.status.idle": "2026-03-09T15:44:57.546808Z",
     "shell.execute_reply": "2026-03-09T15:44:57.545933Z"
    },
    "papermill": {
     "duration": 0.011441,
     "end_time": "2026-03-09T15:44:57.549204",
     "exception": false,
     "start_time": "2026-03-09T15:44:57.537763",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def save_confusion_matrix(epoch, labels, preds, genres, save_path):\n",
    "    cm = confusion_matrix(labels, preds, labels=range(len(genres)))\n",
    "    fig, ax = plt.subplots(figsize=(10, 8))\n",
    "    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genres)\n",
    "    disp.plot(xticks_rotation='vertical', ax=ax, cmap='Blues')\n",
    "    plt.title(f'Confusion Matrix - Epoch {epoch}')\n",
    "    plt.tight_layout()\n",
    "    plt.savefig(save_path)\n",
    "    plt.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a4493925",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-09T15:44:57.555896Z",
     "iopub.status.busy": "2026-03-09T15:44:57.555528Z",
     "iopub.status.idle": "2026-03-09T15:44:57.569438Z",
     "shell.execute_reply": "2026-03-09T15:44:57.568516Z"
    },
    "papermill": {
     "duration": 0.019668,
     "end_time": "2026-03-09T15:44:57.571405",
     "exception": false,
     "start_time": "2026-03-09T15:44:57.551737",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def main(\n",
    "    epochs=15,\n",
    "    batch_size=32,\n",
    "    lr=1e-4,\n",
    "    n_train=2000,\n",
    "    n_val=500,\n",
    "    early_stopping=7,\n",
    "    predict_only=False\n",
    "):\n",
    "    \n",
    "    device = torch.device(\"mps\" if torch.backends.mps.is_available() else \"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "    print(device)\n",
    "\n",
    "    root = \"/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup\" \n",
    "\n",
    "\n",
    "    esc50_root = os.path.join(root, \"ESC-50-master\")\n",
    "    test_csv = os.path.join(root, \"test.csv\")\n",
    "    model_path = \"best_model.pth\"\n",
    "    os.makedirs(\"plots\", exist_ok=True)\n",
    "\n",
    "    if not predict_only:\n",
    "        train_dataset = MessyMashupDataset(root, esc50_root, n_items=n_train)\n",
    "        val_dataset = MessyMashupDataset(root, esc50_root, n_items=n_val)\n",
    "        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n",
    "        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)\n",
    "\n",
    "        model = GenreClassifier(num_classes=len(GENRES)).to(device)\n",
    "        optimizer = optim.Adam(model.parameters(), lr=lr)\n",
    "        criterion = nn.CrossEntropyLoss()\n",
    "        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)\n",
    "\n",
    "        best_f1, patience_counter = 0.0, 0\n",
    "        for epoch in range(epochs):\n",
    "            curr_epoch = epoch + 1\n",
    "            print(f\"\\nEpoch {curr_epoch}/{epochs}\")\n",
    "            _, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)\n",
    "            _, val_f1, v_labels, v_preds = validate(model, val_loader, criterion, device)\n",
    "            \n",
    "            print(f\"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}\")\n",
    "            save_confusion_matrix(curr_epoch, v_labels, v_preds, GENRES, f\"plots/cm_epoch_{curr_epoch}.png\")\n",
    "            \n",
    "            scheduler.step(val_f1)\n",
    "            if val_f1 > best_f1:\n",
    "                best_f1 = val_f1\n",
    "                torch.save(model.state_dict(), model_path)\n",
    "                patience_counter = 0\n",
    "            else:\n",
    "                patience_counter += 1\n",
    "                if patience_counter >= early_stopping: break\n",
    "\n",
    "    # Prediction\n",
    "    if os.path.exists(model_path) and os.path.exists(test_csv):\n",
    "        test_dataset = TestMashupDataset(root, test_csv)\n",
    "        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)\n",
    "        model = GenreClassifier(num_classes=len(GENRES)).to(device)\n",
    "        model.load_state_dict(torch.load(model_path, map_location=device))\n",
    "        model.eval()\n",
    "        \n",
    "        results = []\n",
    "        with torch.no_grad():\n",
    "            for inputs, ids in tqdm(test_loader, desc=\"Predicting\"):\n",
    "                outputs = model(inputs.to(device))\n",
    "                preds = torch.argmax(outputs, dim=1)\n",
    "                for sid, p in zip(ids, preds):\n",
    "                    curr_id = sid.item() if torch.is_tensor(sid) else sid                   \n",
    "                    curr_genre_idx = p.item()                   \n",
    "                    results.append({\n",
    "                        \"id\": curr_id,\n",
    "                        \"genre\": GENRES[curr_genre_idx]\n",
    "                    })\n",
    "        \n",
    "        df = pd.DataFrame(results)\n",
    "        df.to_csv(\"submission.csv\", index=False)\n",
    "        print(\"Success!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "8d7ada89",
   "metadata": {
    "_kg_hide-output": false,
    "execution": {
     "iopub.execute_input": "2026-03-09T15:44:57.577720Z",
     "iopub.status.busy": "2026-03-09T15:44:57.577380Z",
     "iopub.status.idle": "2026-03-09T18:53:03.780299Z",
     "shell.execute_reply": "2026-03-09T18:53:03.779292Z"
    },
    "papermill": {
     "duration": 11286.208409,
     "end_time": "2026-03-09T18:53:03.782305",
     "exception": false,
     "start_time": "2026-03-09T15:44:57.573896",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "cuda\n",
      "\n",
      "Epoch 1/15\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 63/63 [14:25<00:00, 13.74s/it, loss=1.14]\n",
      "Validation: 100%|██████████| 16/16 [02:44<00:00, 10.28s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train F1: 0.3622, Val F1: 0.3624\n",
      "\n",
      "Epoch 2/15\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 63/63 [10:15<00:00,  9.77s/it, loss=0.702]\n",
      "Validation: 100%|██████████| 16/16 [02:22<00:00,  8.94s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train F1: 0.6643, Val F1: 0.6661\n",
      "\n",
      "Epoch 3/15\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 63/63 [09:54<00:00,  9.43s/it, loss=0.479]\n",
      "Validation: 100%|██████████| 16/16 [02:20<00:00,  8.77s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train F1: 0.7563, Val F1: 0.7746\n",
      "\n",
      "Epoch 4/15\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 63/63 [09:39<00:00,  9.20s/it, loss=0.51]\n",
      "Validation: 100%|██████████| 16/16 [02:23<00:00,  8.99s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train F1: 0.8016, Val F1: 0.7536\n",
      "\n",
      "Epoch 5/15\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 63/63 [09:46<00:00,  9.32s/it, loss=0.319]\n",
      "Validation: 100%|██████████| 16/16 [02:21<00:00,  8.83s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train F1: 0.8328, Val F1: 0.8310\n",
      "\n",
      "Epoch 6/15\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 63/63 [09:40<00:00,  9.21s/it, loss=0.436]\n",
      "Validation: 100%|██████████| 16/16 [02:23<00:00,  8.98s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train F1: 0.8412, Val F1: 0.7524\n",
      "\n",
      "Epoch 7/15\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 63/63 [09:41<00:00,  9.24s/it, loss=0.753]\n",
      "Validation: 100%|██████████| 16/16 [02:21<00:00,  8.86s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train F1: 0.8673, Val F1: 0.7813\n",
      "\n",
      "Epoch 8/15\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 63/63 [09:41<00:00,  9.23s/it, loss=0.25]\n",
      "Validation: 100%|██████████| 16/16 [02:18<00:00,  8.65s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train F1: 0.8821, Val F1: 0.7796\n",
      "\n",
      "Epoch 9/15\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 63/63 [09:20<00:00,  8.89s/it, loss=0.252]\n",
      "Validation: 100%|██████████| 16/16 [02:15<00:00,  8.46s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train F1: 0.8798, Val F1: 0.7025\n",
      "\n",
      "Epoch 10/15\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 63/63 [09:15<00:00,  8.82s/it, loss=0.228]\n",
      "Validation: 100%|██████████| 16/16 [02:13<00:00,  8.34s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train F1: 0.9174, Val F1: 0.8906\n",
      "\n",
      "Epoch 11/15\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 63/63 [09:35<00:00,  9.13s/it, loss=0.0936]\n",
      "Validation: 100%|██████████| 16/16 [02:17<00:00,  8.61s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train F1: 0.9270, Val F1: 0.9018\n",
      "\n",
      "Epoch 12/15\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 63/63 [09:43<00:00,  9.27s/it, loss=0.364]\n",
      "Validation: 100%|██████████| 16/16 [02:21<00:00,  8.86s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train F1: 0.9258, Val F1: 0.9333\n",
      "\n",
      "Epoch 13/15\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 63/63 [09:22<00:00,  8.93s/it, loss=0.441]\n",
      "Validation: 100%|██████████| 16/16 [02:15<00:00,  8.49s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train F1: 0.9262, Val F1: 0.9249\n",
      "\n",
      "Epoch 14/15\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 63/63 [09:51<00:00,  9.40s/it, loss=0.0984]\n",
      "Validation: 100%|██████████| 16/16 [02:20<00:00,  8.76s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train F1: 0.9381, Val F1: 0.9486\n",
      "\n",
      "Epoch 15/15\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 63/63 [09:23<00:00,  8.94s/it, loss=0.108]\n",
      "Validation: 100%|██████████| 16/16 [02:16<00:00,  8.54s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train F1: 0.9416, Val F1: 0.8647\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Predicting: 100%|██████████| 95/95 [02:58<00:00,  1.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Success!\n"
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
    "main()"
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
   "duration": 11314.883732,
   "end_time": "2026-03-09T18:53:07.703157",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-03-09T15:44:32.819425",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
