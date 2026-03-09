{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "fd8151c9",
   "metadata": {
    "_cell_guid": "29e03169-88d3-4f63-a7da-f34ae0eb3db0",
    "_uuid": "22e946c8-8912-4165-86f7-6fca72dacd53",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2026-03-09T02:40:32.619864Z",
     "iopub.status.busy": "2026-03-09T02:40:32.619635Z",
     "iopub.status.idle": "2026-03-09T02:40:38.660820Z",
     "shell.execute_reply": "2026-03-09T02:40:38.660209Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 6.047254,
     "end_time": "2026-03-09T02:40:38.662578",
     "exception": false,
     "start_time": "2026-03-09T02:40:32.615324",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Importing Libraries \n",
    "\n",
    "import os\n",
    "import librosa\n",
    "import numpy as np\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from sklearn.model_selection import train_test_split\n",
    "from tqdm import tqdm\n",
    "import concurrent.futures\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score\n",
    "import torchaudio.transforms as T\n",
    "import torchaudio\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ae8ff314",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-09T02:40:38.667003Z",
     "iopub.status.busy": "2026-03-09T02:40:38.666666Z",
     "iopub.status.idle": "2026-03-09T02:40:38.921090Z",
     "shell.execute_reply": "2026-03-09T02:40:38.920331Z"
    },
    "papermill": {
     "duration": 0.258316,
     "end_time": "2026-03-09T02:40:38.922666",
     "exception": false,
     "start_time": "2026-03-09T02:40:38.664350",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running on: cuda\n"
     ]
    }
   ],
   "source": [
    "BASE_INPUT = \"/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup\"\n",
    "BASE_WORKING = \"/kaggle/working\"\n",
    "\n",
    "INPUT_DIR = os.path.join(BASE_INPUT, \"genres_stems\")\n",
    "TEST_CSV = os.path.join(BASE_INPUT, \"test.csv\")\n",
    "MASHUPS_DIR = os.path.join(BASE_INPUT, \"mashups\")\n",
    "\n",
    "WORKING_DIR = os.path.join(BASE_WORKING, \"preprocessed_mel_specs\")\n",
    "TEST_WORKING_DIR = os.path.join(BASE_WORKING, \"preprocessed_test_mel_specs\")\n",
    "CHECKPOINT_PATH = os.path.join(BASE_WORKING, \"best_fusion_resnet.pth\")\n",
    "\n",
    "SR = 22050\n",
    "DURATION = 30\n",
    "N_MELS = 128\n",
    "N_FFT = 2048\n",
    "HOP_LENGTH = 512\n",
    "\n",
    "BATCH_SIZE = 32\n",
    "EPOCHS = 50\n",
    "NUM_WORKERS = min(os.cpu_count(), 8) \n",
    "\n",
    "DEVICE = torch.device(\"mps\" if torch.backends.mps.is_available() else \"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"Running on: {DEVICE}\")\n",
    "\n",
    "try:\n",
    "    if \"soundfile\" in torchaudio.list_audio_backends():\n",
    "        torchaudio.set_audio_backend(\"soundfile\")\n",
    "except:\n",
    "    pass\n",
    "\n",
    "for d in [WORKING_DIR, TEST_WORKING_DIR]:\n",
    "    if not os.path.exists(d):\n",
    "        os.makedirs(d, exist_ok=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "e1c82ace",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-09T02:40:38.927263Z",
     "iopub.status.busy": "2026-03-09T02:40:38.926968Z",
     "iopub.status.idle": "2026-03-09T02:40:38.943528Z",
     "shell.execute_reply": "2026-03-09T02:40:38.942844Z"
    },
    "papermill": {
     "duration": 0.020511,
     "end_time": "2026-03-09T02:40:38.944877",
     "exception": false,
     "start_time": "2026-03-09T02:40:38.924366",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def extract_mel_fusion(genre, track_id, stem_files, input_base_dir):\n",
    "    stems_to_load = ['bass', 'drums', 'other', 'vocals']\n",
    "    track_specs = []\n",
    "    \n",
    "    for stem_type in stems_to_load:\n",
    "        s_file = next((f for f in stem_files if stem_type in f.lower()), None)\n",
    "        \n",
    "        if not s_file:\n",
    "            track_specs.append(np.zeros((N_MELS, DURATION * SR // HOP_LENGTH + 1)))\n",
    "            continue\n",
    "            \n",
    "        path = os.path.join(input_base_dir, genre, track_id, s_file)\n",
    "        try:\n",
    "            y, _ = librosa.load(path, sr=SR, mono=True)\n",
    "            y, _ = librosa.effects.trim(y)\n",
    "            y = librosa.util.fix_length(y, size=SR * DURATION)\n",
    "            \n",
    "            mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)\n",
    "            mel_db = librosa.power_to_db(mel, ref=np.max)\n",
    "            \n",
    "            epsilon = 1e-8\n",
    "            mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + epsilon)\n",
    "            track_specs.append(mel_norm)\n",
    "        except Exception as e:\n",
    "            print(f\"Error loading {path}: {e}\")\n",
    "            track_specs.append(np.zeros((N_MELS, DURATION * SR // HOP_LENGTH + 1)))\n",
    "        \n",
    "    return np.stack(track_specs, axis=0)\n",
    "\n",
    "def worker_preprocess(track_info):\n",
    "    genre, track_id, stems = track_info\n",
    "    save_path = os.path.join(WORKING_DIR, f\"{genre}_{track_id}_fusion.npy\")\n",
    "    \n",
    "    # Check if exists AND is valid (Avoid ValueError: cannot reshape)\n",
    "    if os.path.exists(save_path):\n",
    "        try:\n",
    "            data = np.load(save_path, mmap_mode='r')\n",
    "            expected_shape = (4, N_MELS, DURATION * SR // HOP_LENGTH + 1)\n",
    "            if data.shape == expected_shape:\n",
    "                return False \n",
    "        except:\n",
    "            os.remove(save_path) # Delete corrupted/truncated file\n",
    "            \n",
    "    try:\n",
    "        fusion_spec = extract_mel_fusion(genre, track_id, stems, INPUT_DIR)\n",
    "        np.save(save_path, fusion_spec)\n",
    "        return True\n",
    "    except Exception as e:\n",
    "        print(f\"Error processing {track_id}: {e}\")\n",
    "        return False\n",
    "\n",
    "def worker_test_preprocess(track_info):\n",
    "    track_id = track_info\n",
    "    save_path = os.path.join(TEST_WORKING_DIR, f\"{track_id}_fusion.npy\")\n",
    "    \n",
    "    if os.path.exists(save_path):\n",
    "        try:\n",
    "            data = np.load(save_path, mmap_mode='r')\n",
    "            if data.shape == (4, N_MELS, DURATION * SR // HOP_LENGTH + 1):\n",
    "                return True\n",
    "        except:\n",
    "            os.remove(save_path)\n",
    "    \n",
    "    track_path = os.path.join(MASHUPS_DIR, track_id)\n",
    "    if not os.path.exists(track_path): return False\n",
    "    \n",
    "    try:\n",
    "        stems = [f for f in os.listdir(track_path) if f.endswith(\".wav\")]\n",
    "        if not stems: return False\n",
    "        spec = extract_mel_fusion(\"\", track_id, stems, MASHUPS_DIR.replace(f\"/{track_id}\", \"\"))\n",
    "        np.save(save_path, spec)\n",
    "        return True\n",
    "    except:\n",
    "        return False\n",
    "\n",
    "def run_training_preprocessing():\n",
    "    if not os.path.exists(INPUT_DIR):\n",
    "        print(f\"Input dir not found: {INPUT_DIR}\")\n",
    "        return\n",
    "        \n",
    "    track_tasks = []\n",
    "    genres = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]\n",
    "    for g in genres:\n",
    "        g_path = os.path.join(INPUT_DIR, g)\n",
    "        for tid in os.listdir(g_path):\n",
    "            t_path = os.path.join(g_path, tid)\n",
    "            if os.path.isdir(t_path):\n",
    "                stems = [f for f in os.listdir(t_path) if f.endswith(\".wav\")]\n",
    "                if stems: track_tasks.append((g, tid, stems))\n",
    "\n",
    "    if not track_tasks: return\n",
    "    \n",
    "    print(f\"Preprocessing {len(track_tasks)} tracks...\")\n",
    "    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:\n",
    "        list(tqdm(executor.map(worker_preprocess, track_tasks), total=len(track_tasks), desc=\"Extracting Mel\"))\n",
    "\n",
    "class FusionDataset(Dataset):\n",
    "    def __init__(self, file_list, augment=False):\n",
    "        self.files = file_list\n",
    "        self.augment = augment\n",
    "        # Fixed genres to ensure consistent mapping across train/inference\n",
    "        self.genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']\n",
    "        self.label_map = {g: i for i, g in enumerate(self.genres)}\n",
    "        self.freq_mask = T.FrequencyMasking(30)\n",
    "        self.time_mask = T.TimeMasking(100)\n",
    "\n",
    "    def __len__(self): return len(self.files)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        name = self.files[idx]\n",
    "        path = os.path.join(WORKING_DIR, name)\n",
    "        try:\n",
    "            spec = torch.FloatTensor(np.load(path))\n",
    "            if self.augment:\n",
    "                for i in range(4):\n",
    "                    spec[i:i+1] = self.time_mask(self.freq_mask(spec[i:i+1]))\n",
    "            genre_name = name.split('_')[0]\n",
    "            label = self.label_map.get(genre_name, 0)\n",
    "            return spec, label\n",
    "        except Exception as e:\n",
    "            print(f\"Error loading {name}: {e}\")\n",
    "            return torch.zeros((4, 128, 1292)), 0\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0858dd53",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-09T02:40:38.949369Z",
     "iopub.status.busy": "2026-03-09T02:40:38.948903Z",
     "iopub.status.idle": "2026-03-09T02:40:38.957496Z",
     "shell.execute_reply": "2026-03-09T02:40:38.956844Z"
    },
    "papermill": {
     "duration": 0.012333,
     "end_time": "2026-03-09T02:40:38.958833",
     "exception": false,
     "start_time": "2026-03-09T02:40:38.946500",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "class ResBlock(nn.Module):\n",
    "    def __init__(self, in_ch, out_ch, stride=1):\n",
    "        super().__init__()\n",
    "        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU())\n",
    "        self.conv2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch))\n",
    "        self.shortcut = nn.Sequential()\n",
    "        if stride != 1 or in_ch != out_ch:\n",
    "            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride), nn.BatchNorm2d(out_ch))\n",
    "\n",
    "    def forward(self, x):\n",
    "        return torch.relu(self.conv2(self.conv1(x)) + self.shortcut(x))\n",
    "\n",
    "class StemExpert(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.net = nn.Sequential(\n",
    "            ResBlock(1, 32, stride=2),\n",
    "            ResBlock(32, 64, stride=2),\n",
    "            ResBlock(64, 128, stride=4),\n",
    "            nn.AdaptiveAvgPool2d(1)\n",
    "        )\n",
    "\n",
    "    def forward(self, x):\n",
    "        return self.net(x).flatten(1) # [Batch, 128]\n",
    "\n",
    "class FusionResNet(nn.Module):\n",
    "    def __init__(self, num_classes):\n",
    "        super().__init__()\n",
    "        self.experts = nn.ModuleList([StemExpert() for _ in range(4)])\n",
    "        self.attention = nn.Parameter(torch.ones(4))\n",
    "        self.classifier = nn.Sequential(\n",
    "            nn.Dropout(0.5),\n",
    "            nn.Linear(128 * 4, 128),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(128, num_classes)\n",
    "        )\n",
    "\n",
    "    def forward(self, x):\n",
    "        # x: [Batch, 4, 128, T]\n",
    "        features = [self.experts[i](x[:, i:i+1, :, :]) for i in range(4)]\n",
    "        attn_weights = torch.softmax(self.attention, dim=0)\n",
    "        fused = torch.cat([f * attn_weights[i] for i, f in enumerate(features)], dim=1)\n",
    "        return self.classifier(fused)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "4164e151",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-09T02:40:38.962907Z",
     "iopub.status.busy": "2026-03-09T02:40:38.962671Z",
     "iopub.status.idle": "2026-03-09T02:40:38.969924Z",
     "shell.execute_reply": "2026-03-09T02:40:38.969267Z"
    },
    "papermill": {
     "duration": 0.01102,
     "end_time": "2026-03-09T02:40:38.971420",
     "exception": false,
     "start_time": "2026-03-09T02:40:38.960400",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def train_one_epoch(model, loader, optimizer, criterion):\n",
    "    model.train()\n",
    "    total_loss, all_preds, all_labels = 0, [], []\n",
    "    for inputs, labels in tqdm(loader, desc=\"Training\"):\n",
    "        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)\n",
    "        optimizer.zero_grad()\n",
    "        outputs = model(inputs)\n",
    "        loss = criterion(outputs, labels)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        \n",
    "        total_loss += loss.item()\n",
    "        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())\n",
    "        all_labels.extend(labels.cpu().numpy())\n",
    "    return total_loss / len(loader), f1_score(all_labels, all_preds, average='macro')\n",
    "\n",
    "def validate(model, loader, criterion):\n",
    "    model.eval()\n",
    "    total_loss, all_preds, all_labels = 0, [], []\n",
    "    with torch.no_grad():\n",
    "        for inputs, labels in loader:\n",
    "            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)\n",
    "            outputs = model(inputs)\n",
    "            loss = criterion(outputs, labels)\n",
    "            total_loss += loss.item()\n",
    "            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())\n",
    "            all_labels.extend(labels.cpu().numpy())\n",
    "    return total_loss / len(loader), f1_score(all_labels, all_preds, average='macro'), all_preds, all_labels\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "e721ec13",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-09T02:40:38.975864Z",
     "iopub.status.busy": "2026-03-09T02:40:38.975360Z",
     "iopub.status.idle": "2026-03-09T02:40:38.982697Z",
     "shell.execute_reply": "2026-03-09T02:40:38.982173Z"
    },
    "papermill": {
     "duration": 0.010973,
     "end_time": "2026-03-09T02:40:38.983945",
     "exception": false,
     "start_time": "2026-03-09T02:40:38.972972",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def run_inference(model, genres):\n",
    "    if not os.path.exists(TEST_CSV): \n",
    "        print(f\"Error: {TEST_CSV} not found.\")\n",
    "        return\n",
    "\n",
    "    test_df = pd.read_csv(TEST_CSV)\n",
    "    track_ids = test_df['id'].astype(str).str.zfill(4).tolist()\n",
    "    \n",
    "    print(\"Starting Test Preprocessing...\")\n",
    "    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:\n",
    "        list(tqdm(executor.map(worker_test_preprocess, track_ids), total=len(track_ids), desc=\"Test Prep\"))\n",
    "\n",
    "    model.eval()\n",
    "    results = []\n",
    "    \n",
    "    with torch.no_grad():\n",
    "        for tid in tqdm(track_ids, desc=\"Final Predictions\"):\n",
    "            spec_path = os.path.join(TEST_WORKING_DIR, f\"{tid}_fusion.npy\")\n",
    "            \n",
    "            try:\n",
    "                if not os.path.exists(spec_path):\n",
    "                    raise FileNotFoundError\n",
    "                \n",
    "                spec_np = np.load(spec_path)\n",
    "                if spec_np.shape != (4, 128, 1292):\n",
    "                    raise ValueError\n",
    "                \n",
    "                spec = torch.from_numpy(spec_np).float().unsqueeze(0).to(DEVICE)\n",
    "                logits = model(spec)\n",
    "                pred = torch.argmax(logits, dim=1).item()\n",
    "                results.append({'id': str(tid), 'genre': genres[pred]})\n",
    "            except:\n",
    "                # Fallback to avoid KeyError 'id' if some files are missing\n",
    "                results.append({'id': str(tid), 'genre': 'rock'})\n",
    "\n",
    "    sub_df = pd.DataFrame(results)\n",
    "    sub_df['id'] = sub_df['id'].astype(str).str.zfill(4)\n",
    "\n",
    "    # Ensure submission matches the exact IDs in the test set\n",
    "    final_sub = pd.read_csv(TEST_CSV)[['id']]\n",
    "    final_sub['id'] = final_sub['id'].astype(str).str.zfill(4)\n",
    "    final_sub = final_sub.merge(sub_df, on='id', how='left')\n",
    "    final_sub['genre'] = final_sub['genre'].fillna('rock')\n",
    "\n",
    "    final_sub.to_csv(\"submission.csv\", index=False)\n",
    "    print(\"Done! Submission saved to submission.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "5d491c46",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-09T02:40:38.988206Z",
     "iopub.status.busy": "2026-03-09T02:40:38.987873Z",
     "iopub.status.idle": "2026-03-09T02:40:38.994425Z",
     "shell.execute_reply": "2026-03-09T02:40:38.993863Z"
    },
    "papermill": {
     "duration": 0.010152,
     "end_time": "2026-03-09T02:40:38.995710",
     "exception": false,
     "start_time": "2026-03-09T02:40:38.985558",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def main():\n",
    "    run_training_preprocessing()\n",
    "    \n",
    "    all_files = [f for f in os.listdir(WORKING_DIR) if f.endswith('_fusion.npy')]\n",
    "    if not all_files:\n",
    "        print(\"No training data found.\")\n",
    "        return\n",
    "\n",
    "    train_files, val_files = train_test_split(all_files, test_size=0.15, random_state=42)\n",
    "    train_loader = DataLoader(FusionDataset(train_files, augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)\n",
    "    val_loader = DataLoader(FusionDataset(val_files, augment=False), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)\n",
    "    \n",
    "    genre_list = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']\n",
    "    model = FusionResNet(len(genre_list)).to(DEVICE)\n",
    "    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)\n",
    "    optimizer = optim.AdamW(model.parameters(), lr=0.001)\n",
    "    \n",
    "    if os.path.exists(CHECKPOINT_PATH):\n",
    "        print(f\"Loading checkpoint: {CHECKPOINT_PATH}\")\n",
    "        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))\n",
    "\n",
    "    # Training Loop \n",
    "    if EPOCHS > 0:\n",
    "        best_f1 = 0.0\n",
    "        for epoch in range(EPOCHS):\n",
    "            t_loss, t_f1 = train_one_epoch(model, train_loader, optimizer, criterion)\n",
    "            v_loss, v_f1, _, _ = validate(model, val_loader, criterion)\n",
    "            print(f\"Epoch {epoch+1}/{EPOCHS}: Train F1={t_f1:.4f} | Val F1={v_f1:.4f}\")\n",
    "            if v_f1 > best_f1:\n",
    "                best_f1 = v_f1\n",
    "                torch.save(model.state_dict(), CHECKPOINT_PATH)\n",
    "\n",
    "    # Inference\n",
    "    if os.path.exists(CHECKPOINT_PATH):\n",
    "        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))\n",
    "    run_inference(model, genre_list)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "7a5709c2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-09T02:40:39.000044Z",
     "iopub.status.busy": "2026-03-09T02:40:38.999619Z",
     "iopub.status.idle": "2026-03-09T03:02:50.839874Z",
     "shell.execute_reply": "2026-03-09T03:02:50.839126Z"
    },
    "papermill": {
     "duration": 1331.843941,
     "end_time": "2026-03-09T03:02:50.841352",
     "exception": false,
     "start_time": "2026-03-09T02:40:38.997411",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Preprocessing 1000 tracks...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Extracting Mel: 100%|██████████| 1000/1000 [05:47<00:00,  2.87it/s]\n",
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.47it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50: Train F1=0.3152 | Val F1=0.2848\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:17<00:00,  1.57it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 2/50: Train F1=0.5081 | Val F1=0.1296\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:17<00:00,  1.56it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 3/50: Train F1=0.5809 | Val F1=0.4225\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:17<00:00,  1.55it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 4/50: Train F1=0.6288 | Val F1=0.1744\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:17<00:00,  1.54it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 5/50: Train F1=0.6587 | Val F1=0.5361\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:17<00:00,  1.51it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 6/50: Train F1=0.6790 | Val F1=0.3602\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.49it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 7/50: Train F1=0.6870 | Val F1=0.6717\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.49it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 8/50: Train F1=0.7059 | Val F1=0.4435\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 9/50: Train F1=0.7332 | Val F1=0.4682\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:17<00:00,  1.51it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 10/50: Train F1=0.7446 | Val F1=0.6515\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 11/50: Train F1=0.7288 | Val F1=0.5528\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.49it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 12/50: Train F1=0.7506 | Val F1=0.5726\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 13/50: Train F1=0.7595 | Val F1=0.5942\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 14/50: Train F1=0.8055 | Val F1=0.5592\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 15/50: Train F1=0.7736 | Val F1=0.3113\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 16/50: Train F1=0.7994 | Val F1=0.6226\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 17/50: Train F1=0.7941 | Val F1=0.6214\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 18/50: Train F1=0.7914 | Val F1=0.6986\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.49it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 19/50: Train F1=0.8140 | Val F1=0.4275\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 20/50: Train F1=0.8275 | Val F1=0.7300\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 21/50: Train F1=0.8393 | Val F1=0.5501\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.49it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 22/50: Train F1=0.8286 | Val F1=0.6950\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.49it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 23/50: Train F1=0.8444 | Val F1=0.6403\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.49it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 24/50: Train F1=0.8434 | Val F1=0.7769\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.49it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 25/50: Train F1=0.8717 | Val F1=0.5558\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 26/50: Train F1=0.8330 | Val F1=0.7218\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 27/50: Train F1=0.8507 | Val F1=0.7131\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 28/50: Train F1=0.8701 | Val F1=0.6355\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.49it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 29/50: Train F1=0.8669 | Val F1=0.6574\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 30/50: Train F1=0.8950 | Val F1=0.7835\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 31/50: Train F1=0.8737 | Val F1=0.6786\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.49it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 32/50: Train F1=0.8832 | Val F1=0.6482\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.49it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 33/50: Train F1=0.8882 | Val F1=0.5178\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 34/50: Train F1=0.9018 | Val F1=0.5967\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 35/50: Train F1=0.9040 | Val F1=0.6606\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.49it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 36/50: Train F1=0.8945 | Val F1=0.6037\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 37/50: Train F1=0.9065 | Val F1=0.5564\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 38/50: Train F1=0.9133 | Val F1=0.7447\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.49it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 39/50: Train F1=0.9177 | Val F1=0.6246\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 40/50: Train F1=0.9240 | Val F1=0.7364\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 41/50: Train F1=0.9234 | Val F1=0.7646\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 42/50: Train F1=0.9306 | Val F1=0.7084\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:17<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 43/50: Train F1=0.9138 | Val F1=0.6179\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 44/50: Train F1=0.9332 | Val F1=0.6443\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 45/50: Train F1=0.9391 | Val F1=0.4619\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.49it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 46/50: Train F1=0.9321 | Val F1=0.5866\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 47/50: Train F1=0.9311 | Val F1=0.5888\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.49it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 48/50: Train F1=0.9438 | Val F1=0.5356\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.49it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 49/50: Train F1=0.9321 | Val F1=0.4552\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training: 100%|██████████| 27/27 [00:18<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 50/50: Train F1=0.9482 | Val F1=0.3364\n",
      "Starting Test Preprocessing...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test Prep: 100%|██████████| 3020/3020 [00:00<00:00, 4781.34it/s]\n",
      "Final Predictions: 100%|██████████| 3020/3020 [00:00<00:00, 194550.56it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Done! Submission saved to submission.csv\n"
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
   "duration": 1343.097725,
   "end_time": "2026-03-09T03:02:53.347533",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-03-09T02:40:30.249808",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
