{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "33fe7500",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2026-02-19T07:29:09.991602Z",
     "iopub.status.busy": "2026-02-19T07:29:09.991192Z",
     "iopub.status.idle": "2026-02-19T07:29:14.152488Z",
     "shell.execute_reply": "2026-02-19T07:29:14.151335Z"
    },
    "papermill": {
     "duration": 4.168732,
     "end_time": "2026-02-19T07:29:14.154934",
     "exception": false,
     "start_time": "2026-02-19T07:29:09.986202",
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
    "import librosa\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from pathlib import Path\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import f1_score\n",
    "from catboost import CatBoostClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "834713ef",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-19T07:29:14.162049Z",
     "iopub.status.busy": "2026-02-19T07:29:14.161528Z",
     "iopub.status.idle": "2026-02-19T07:29:14.168760Z",
     "shell.execute_reply": "2026-02-19T07:29:14.167608Z"
    },
    "papermill": {
     "duration": 0.013545,
     "end_time": "2026-02-19T07:29:14.171004",
     "exception": false,
     "start_time": "2026-02-19T07:29:14.157459",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Configuration\n",
    "\n",
    "DATA_SEED = 67\n",
    "TRAINING_SEED = 1234\n",
    "SR = 22050\n",
    "DURATION = 5.0\n",
    "TOP_DB = 20\n",
    "DATA_ROOT = '/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup'\n",
    "GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']\n",
    "STEMS = {'drums': 'drums.wav', 'vocals': 'vocals.wav', 'bass': 'bass.wav', 'other': 'other.wav'}\n",
    "STEM_KEYS = ['drums', 'vocals', 'bass', 'other']\n",
    "\n",
    "random.seed(DATA_SEED)\n",
    "np.random.seed(DATA_SEED)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "177af29f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-19T07:29:14.177699Z",
     "iopub.status.busy": "2026-02-19T07:29:14.177241Z",
     "iopub.status.idle": "2026-02-19T07:29:14.188892Z",
     "shell.execute_reply": "2026-02-19T07:29:14.187669Z"
    },
    "papermill": {
     "duration": 0.017936,
     "end_time": "2026-02-19T07:29:14.191281",
     "exception": false,
     "start_time": "2026-02-19T07:29:14.173345",
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
   "id": "cd3be85b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-19T07:29:14.198223Z",
     "iopub.status.busy": "2026-02-19T07:29:14.197559Z",
     "iopub.status.idle": "2026-02-19T07:29:14.206063Z",
     "shell.execute_reply": "2026-02-19T07:29:14.204965Z"
    },
    "papermill": {
     "duration": 0.014153,
     "end_time": "2026-02-19T07:29:14.208117",
     "exception": false,
     "start_time": "2026-02-19T07:29:14.193964",
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
   "id": "9f07fe60",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-19T07:29:14.214142Z",
     "iopub.status.busy": "2026-02-19T07:29:14.213791Z",
     "iopub.status.idle": "2026-02-19T07:29:14.222128Z",
     "shell.execute_reply": "2026-02-19T07:29:14.220811Z"
    },
    "papermill": {
     "duration": 0.014511,
     "end_time": "2026-02-19T07:29:14.224977",
     "exception": false,
     "start_time": "2026-02-19T07:29:14.210466",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Feature Extraction\n",
    "\n",
    "def extract_mashup_features(dataset_dict, num_samples=100):\n",
    "    features = []\n",
    "    for genre in GENRES:\n",
    "        for _ in range(num_samples):\n",
    "            mix = np.zeros(int(SR * DURATION))\n",
    "            for key in STEM_KEYS:\n",
    "                path = random.choice(dataset_dict[genre][key])\n",
    "                y, _ = librosa.load(path, sr=SR, duration=DURATION)\n",
    "                mix += y\n",
    "            \n",
    "            mfcc = np.mean(librosa.feature.mfcc(y=mix, sr=SR, n_mfcc=13), axis=1)\n",
    "            centroid = np.mean(librosa.feature.spectral_centroid(y=mix, sr=SR))\n",
    "            \n",
    "            row = {f'mfcc_{i}': val for i, val in enumerate(mfcc)}\n",
    "            row.update({'centroid': centroid, 'label': genre})\n",
    "            features.append(row)\n",
    "            \n",
    "    return pd.DataFrame(features)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "9d82629d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-19T07:29:14.231687Z",
     "iopub.status.busy": "2026-02-19T07:29:14.231323Z",
     "iopub.status.idle": "2026-02-19T07:37:34.952559Z",
     "shell.execute_reply": "2026-02-19T07:37:34.950925Z"
    },
    "papermill": {
     "duration": 500.732397,
     "end_time": "2026-02-19T07:37:34.959818",
     "exception": false,
     "start_time": "2026-02-19T07:29:14.227421",
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
      "Genre: rock       | Train:  83 | Val:  17\n",
      "Total files with >= 5s silence: 2094\n",
      "\n",
      "Training...\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<catboost.core.CatBoostClassifier at 0x7f98983a7fb0>"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Model Building Pipeline\n",
    "\n",
    "tr, val = build_dataset(DATA_ROOT)\n",
    "\n",
    "df_silence = find_long_silences(tr)\n",
    "print(f\"Total files with >= 5s silence: {len(df_silence)}\")\n",
    "\n",
    "print(\"\\nTraining...\")\n",
    "df_train = extract_mashup_features(tr, num_samples=50)\n",
    "X = df_train.drop('label', axis=1)\n",
    "le = LabelEncoder()\n",
    "y = le.fit_transform(df_train['label'])\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=TRAINING_SEED)\n",
    "\n",
    "clf = CatBoostClassifier(iterations=300, depth=6, verbose=0, loss_function='MultiClass')\n",
    "clf.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "dfc19128",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-19T07:37:34.970382Z",
     "iopub.status.busy": "2026-02-19T07:37:34.968874Z",
     "iopub.status.idle": "2026-02-19T07:39:32.295792Z",
     "shell.execute_reply": "2026-02-19T07:39:32.294634Z"
    },
    "papermill": {
     "duration": 117.33577,
     "end_time": "2026-02-19T07:39:32.298765",
     "exception": false,
     "start_time": "2026-02-19T07:37:34.962995",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "--- Processing 3020 Test Mashups ---\n",
      "Processed 100/3020 files...\n",
      "Processed 200/3020 files...\n",
      "Processed 300/3020 files...\n",
      "Processed 400/3020 files...\n",
      "Processed 500/3020 files...\n",
      "Processed 600/3020 files...\n",
      "Processed 700/3020 files...\n",
      "Processed 800/3020 files...\n",
      "Processed 900/3020 files...\n",
      "Processed 1000/3020 files...\n",
      "Processed 1100/3020 files...\n",
      "Processed 1200/3020 files...\n",
      "Processed 1300/3020 files...\n",
      "Processed 1400/3020 files...\n",
      "Processed 1500/3020 files...\n",
      "Processed 1600/3020 files...\n",
      "Processed 1700/3020 files...\n",
      "Processed 1800/3020 files...\n",
      "Processed 1900/3020 files...\n",
      "Processed 2000/3020 files...\n",
      "Processed 2100/3020 files...\n",
      "Processed 2200/3020 files...\n",
      "Processed 2300/3020 files...\n",
      "Processed 2400/3020 files...\n",
      "Processed 2500/3020 files...\n",
      "Processed 2600/3020 files...\n",
      "Processed 2700/3020 files...\n",
      "Processed 2800/3020 files...\n",
      "Processed 2900/3020 files...\n",
      "Processed 3000/3020 files...\n"
     ]
    }
   ],
   "source": [
    "test_df = pd.read_csv('/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/test.csv') \n",
    "test_features = []\n",
    "\n",
    "print(f\"--- Processing {len(test_df)} Test Mashups ---\")\n",
    "\n",
    "for index, row in test_df.iterrows():\n",
    "    file_path = os.path.join(DATA_ROOT, row['filename'])\n",
    "    \n",
    "    try:\n",
    "        y, _ = librosa.load(file_path, sr=SR, duration=DURATION)\n",
    "        \n",
    "        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13), axis=1)\n",
    "        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=SR))\n",
    "        \n",
    "        test_features.append(list(mfcc) + [centroid])\n",
    "        \n",
    "    except Exception as e:\n",
    "        print(f\"Error processing {file_path}: {e}\")\n",
    "        test_features.append([0]*14)\n",
    "\n",
    "    if (index + 1) % 100 == 0:\n",
    "        print(f\"Processed {index + 1}/{len(test_df)} files...\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "31d4d310",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-19T07:39:32.317651Z",
     "iopub.status.busy": "2026-02-19T07:39:32.317222Z",
     "iopub.status.idle": "2026-02-19T07:39:32.406300Z",
     "shell.execute_reply": "2026-02-19T07:39:32.405202Z"
    },
    "papermill": {
     "duration": 0.101808,
     "end_time": "2026-02-19T07:39:32.408532",
     "exception": false,
     "start_time": "2026-02-19T07:39:32.306724",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Success!\n",
      "   id   genre\n",
      "0   1     pop\n",
      "1   2  reggae\n",
      "2   3   disco\n",
      "3   4    jazz\n",
      "4   5  reggae\n"
     ]
    }
   ],
   "source": [
    "# Prediction and Submmission\n",
    "\n",
    "X_test_final = pd.DataFrame(test_features, columns=[f'mfcc_{i}' for i in range(13)] + ['centroid'])\n",
    "test_preds = clf.predict(X_test_final)\n",
    "\n",
    "\n",
    "predicted_genres = le.inverse_transform(test_preds.flatten())\n",
    "\n",
    "submission_df = pd.DataFrame({\n",
    "    'id': test_df['id'], \n",
    "    'genre': predicted_genres\n",
    "})\n",
    "\n",
    "submission_df.to_csv('submission.csv', index=False)\n",
    "\n",
    "print(\"\\nSuccess!\")\n",
    "print(submission_df.head())"
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
   "duration": 628.102334,
   "end_time": "2026-02-19T07:39:34.144888",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-02-19T07:29:06.042554",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
