import os
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.shap_utils import (
    pick_positive_class_index,
    select_pos_shap,
    select_base_value
)


def run(model, X_test, results_df, output_dir="results/shap"):

    os.makedirs(output_dir, exist_ok=True)
    errors_dir = os.path.join(output_dir, "errors")
    corrects_dir = os.path.join(output_dir, "corrects")

    os.makedirs(errors_dir, exist_ok=True)
    os.makedirs(corrects_dir, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_test)

    pos_idx = pick_positive_class_index(model)
    sv_pos = select_pos_shap(shap_values_raw, X_test.shape[1], pos_idx)
    base_val = select_base_value(explainer.expected_value, pos_idx)

    # ===== GLOBAL =====
    plt.figure()
    shap.summary_plot(sv_pos, X_test, feature_names=X_test.columns, show=False)
    plt.savefig(os.path.join(output_dir, "global_summary.png"))
    plt.close()

    # ===== PER SAMPLE =====
    for i in range(len(X_test)):
        exp = shap.Explanation(
            values=sv_pos[i],
            base_values=base_val,
            data=X_test.iloc[i].to_numpy(),
            feature_names=list(X_test.columns)
        )

        folder = corrects_dir if results_df.iloc[i]["status"] == "certo" else errors_dir

        plt.figure()
        shap.plots.waterfall(exp, show=False)
        plt.savefig(os.path.join(folder, f"{i}_waterfall.png"))
        plt.close()

    print(f"[INFO] SHAP results saved to {output_dir}")
