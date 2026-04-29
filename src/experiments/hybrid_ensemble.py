import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from tensorflow.keras.models import load_model

from src.utils.ensemble import balance_train, bias_correction


def load_maps(video, label, npz_folder):
    base = video.replace(".mp4", "")
    suffix = "-real.npz" if int(label) == 0 else "-fake.npz"
    path = os.path.join(npz_folder, base + suffix)

    if not os.path.exists(path):
        return np.zeros((1, 128, 64, 1), dtype=np.float32)

    maps = np.load(path)["maps"].astype(np.float32) / 255.0

    if maps.ndim == 2:
        maps = maps[None, ..., None]
    elif maps.ndim == 3:
        maps = maps[..., None]

    return maps


def cnn_predict_mean(model, maps_list):
    out = np.zeros(len(maps_list), dtype=np.float32)

    for i, v in enumerate(maps_list):
        pv = model.predict(v, verbose=0)
        out[i] = float(np.mean(pv))

    return out


def run(csv_path, npz_folder, cnn_path):

    df = pd.read_csv(csv_path)

    X = df.drop(columns=["video", "label"])
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Preload maps
    ppg = {}
    for i, r in tqdm(df.iterrows(), total=len(df)):
        ppg[i] = load_maps(r["video"], r["label"], npz_folder)

    # OOF
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cnn_oof = np.zeros(len(X_train))
    rf_oof = np.zeros(len(X_train))

    for fold, (tr, val) in enumerate(skf.split(X_train, y_train)):
        print(f"[Fold {fold+1}]")

        X_tr, y_tr = X_train.iloc[tr], y_train.iloc[tr]
        X_val = X_train.iloc[val]

        Xb, yb = balance_train(X_tr, y_tr)

        rf = RandomForestClassifier(n_estimators=200, max_depth=10)
        rf.fit(Xb, yb)

        rf_oof[val] = rf.predict_proba(X_val)[:, 1]

        cnn = load_model(cnn_path)

        idx = X_train.index[val]
        maps = [ppg[i] for i in idx]

        cnn_oof[val] = cnn_predict_mean(cnn, maps)

    # Simple best threshold (you can reuse your sweep)
    best_thr = 0.5

    rf_final = RandomForestClassifier(n_estimators=200, max_depth=10)
    Xb, yb = balance_train(X_train, y_train)
    rf_final.fit(Xb, yb)

    cnn_final = load_model(cnn_path)

    test_maps = [ppg[i] for i in X_test.index]

    cnn_p = cnn_predict_mean(cnn_final, test_maps)
    rf_p = rf_final.predict_proba(X_test)[:, 1]

    proba, pred, fixed = bias_correction(
        cnn_p,
        rf_p,
        thr_cnn=best_thr,
        margin=0.1,
        real_rf_max=0.3,
        fake_rf_min=0.7
    )

    print("\n===== RESULTS =====")

    def summarize(name, y_true, p, pr):
        return (
            roc_auc_score(y_true, p),
            precision_score(y_true, pr),
            recall_score(y_true, pr),
            f1_score(y_true, pr)
        )

    print("CNN:", summarize("cnn", y_test, cnn_p, (cnn_p > 0.5)))
    print("RF :", summarize("rf", y_test, rf_p, (rf_p > 0.5)))
    print("HYB:", summarize("hyb", y_test, proba, pred))
