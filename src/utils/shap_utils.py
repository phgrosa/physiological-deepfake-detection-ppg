import numpy as np


def pick_positive_class_index(model):
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        return classes.index(1) if 1 in classes else len(classes) - 1
    return -1


def select_pos_shap(shap_vals, n_features, pos_idx):

    def _extract(arr):
        arr = np.asarray(arr)

        if arr.ndim == 2 and arr.shape[1] == n_features:
            return arr

        if arr.ndim == 3:
            if arr.shape[0] <= 10 and arr.shape[2] == n_features:
                return arr[pos_idx if pos_idx >= 0 else -1, :, :]

            if arr.shape[2] <= 10 and arr.shape[1] == n_features:
                return arr[:, :, pos_idx if pos_idx >= 0 else -1]

        return None

    if isinstance(shap_vals, list):
        for sv in shap_vals:
            res = _extract(sv)
            if res is not None:
                return res
        raise ValueError("Could not align SHAP values")

    res = _extract(shap_vals)
    if res is not None:
        return res

    raise ValueError("Invalid SHAP format")


def select_base_value(expected_value, pos_idx):
    ev = np.array(expected_value)

    if ev.ndim == 0:
        return float(ev)

    flat = ev.reshape(-1)
    return float(flat[pos_idx] if pos_idx < len(flat) else flat[-1])
