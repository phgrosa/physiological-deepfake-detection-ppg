import argparse
import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    recall_score
)
from sklearn.utils import class_weight

from src.utils.cnn_models import build_fakecatcher_cnn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train FakeCatcher CNN using saved G-PPG and C-PPG maps"
    )

    parser.add_argument("--maps_dir", required=True, help="Directory containing spatial/ and spectral/ folders")
    parser.add_argument("--models_dir", required=True, help="Directory to save trained models")
    parser.add_argument("--target_shape", type=int, nargs=3, default=[64, 64, 3])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    return parser.parse_args()


def build_labels_dict(npz_paths):
    labels = {}

    for path in npz_paths:
        name = os.path.splitext(os.path.basename(path))[0]

        if name.endswith("-real"):
            labels[name] = 0
        elif name.endswith("-fake"):
            labels[name] = 1
        else:
            print(f"[WARNING] File without '-real' or '-fake' suffix: {name}")

    return labels


def load_dataset(npz_list, labels_dict, target_shape=(64, 64, 3)):
    X, y = [], []

    for npz_path in npz_list:
        name = os.path.splitext(os.path.basename(npz_path))[0]

        if name not in labels_dict:
            print(f"[WARNING] {name} not found in labels dictionary. Skipping.")
            continue

        try:
            maps = np.load(npz_path, allow_pickle=True)["maps"]
        except Exception as exc:
            print(f"[ERROR] Could not load {npz_path}: {exc}")
            continue

        for m in maps:
            m = np.asarray(m)

            if m.ndim == 2:
                m = np.expand_dims(m, axis=-1)
                m = np.repeat(m, 3, axis=-1)

            m = tf.image.resize(m, target_shape[:2]).numpy()

            X.append(m)
            y.append(labels_dict[name])

    if len(X) == 0:
        raise ValueError("No valid maps were loaded.")

    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int32)


def augment_maps(x, y):
    def augment(img):
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        return img

    x = tf.cond(tf.equal(y, 0), lambda: augment(x), lambda: x)
    return x, y


def make_tf_dataset(X, y, batch_size=32, shuffle=True, augment=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        ds = ds.shuffle(len(y))

    if augment:
        ds = ds.map(augment_maps, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def compute_class_weights(y):
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )

    return {int(cls): float(w) for cls, w in zip(np.unique(y), weights)}


def oversample_real_class(X_train, y_train):
    real_mask = y_train == 0

    X_train = np.concatenate([X_train, X_train[real_mask]])
    y_train = np.concatenate([y_train, y_train[real_mask]])

    return X_train, y_train


def evaluate_model(model, ds_test, y_test, title):
    y_pred = (model.predict(ds_test) > 0.5).astype(int).flatten()

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.show()

    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"], digits=4))
    print(f"Recall Real: {recall_score(y_test, y_pred, pos_label=0):.4f}")
    print(f"Recall Fake: {recall_score(y_test, y_pred, pos_label=1):.4f}")


def train_one_map_type(
    map_type,
    npz_paths,
    labels_dict,
    args
):
    print(f"\n[INFO] Loading {map_type} maps...")

    X, y = load_dataset(
        npz_paths,
        labels_dict,
        target_shape=tuple(args.target_shape)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state
    )

    X_train, y_train = oversample_real_class(X_train, y_train)

    train_ds = make_tf_dataset(
        X_train,
        y_train,
        batch_size=args.batch_size,
        shuffle=True,
        augment=True
    )

    test_ds = make_tf_dataset(
        X_test,
        y_test,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False
    )

    class_weights = compute_class_weights(y_train)

    model = build_fakecatcher_cnn(input_shape=X_train.shape[1:])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print(f"[INFO] Training {map_type} CNN...")

    model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=test_ds,
        class_weight=class_weights,
        verbose=1
    )

    os.makedirs(args.models_dir, exist_ok=True)

    model_path = os.path.join(args.models_dir, f"fakecatcher_{map_type}.keras")
    model.save(model_path)

    print(f"[INFO] Model saved to: {model_path}")

    evaluate_model(
        model,
        test_ds,
        y_test,
        title=f"Confusion Matrix - {map_type.capitalize()}"
    )


def main():
    args = parse_args()

    spatial_paths = glob.glob(os.path.join(args.maps_dir, "spatial", "*.npz"))
    spectral_paths = glob.glob(os.path.join(args.maps_dir, "spectral", "*.npz"))

    all_paths = spatial_paths + spectral_paths
    labels_dict = build_labels_dict(all_paths)

    train_one_map_type(
        map_type="spatial",
        npz_paths=spatial_paths,
        labels_dict=labels_dict,
        args=args
    )

    train_one_map_type(
        map_type="spectral",
        npz_paths=spectral_paths,
        labels_dict=labels_dict,
        args=args
    )


if __name__ == "__main__":
    main()