import numpy as np


def balance_train(X, y, random_state=42):
    df = X.copy()
    df["label"] = y.values

    real = df[df.label == 0]
    fake = df[df.label == 1]

    if len(real) < len(fake):
        real = real.sample(len(fake), replace=True, random_state=random_state)
    else:
        fake = fake.sample(len(real), replace=True, random_state=random_state)

    df = df.sample(frac=1, random_state=random_state)
    return df.drop(columns=["label"]), df["label"]


def bias_correction(
    cnn_p,
    rf_p,
    thr_cnn,
    margin,
    real_rf_max,
    fake_rf_min,
    allow_fp_fix=True,
    allow_fn_fix=False
):
    cnn_p = np.asarray(cnn_p)
    rf_p = np.asarray(rf_p)

    cnn_pred = (cnn_p > thr_cnn).astype(int)

    low = thr_cnn - margin
    high = thr_cnn + margin

    in_margin = (cnn_p >= low) & (cnn_p <= high)

    rf_real_conf = rf_p <= real_rf_max
    rf_fake_conf = rf_p >= fake_rf_min

    final_pred = cnn_pred.copy()
    final_proba = cnn_p.copy()

    fixed = np.zeros_like(cnn_pred, dtype=bool)

    if allow_fp_fix:
        mask = (cnn_pred == 1) & in_margin & rf_real_conf
        final_pred[mask] = 0
        final_proba[mask] = rf_p[mask]
        fixed |= mask

    if allow_fn_fix:
        mask = (cnn_pred == 0) & in_margin & rf_fake_conf
        final_pred[mask] = 1
        final_proba[mask] = rf_p[mask]
        fixed |= mask

    return final_proba, final_pred, fixed
