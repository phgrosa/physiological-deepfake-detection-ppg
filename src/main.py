# src/main.py

import argparse

from src.experiments import (
    ablation_rf,
    extract_features,
    fakecatcher_ppg_cnn,
    hybrid_ensemble,
)


def main():
    parser = argparse.ArgumentParser(
        description="Deepfake detection research pipeline"
    )

    parser.add_argument(
        "--exp",
        required=True,
        choices=[
            "extract_features",
            "ablation_rf",
            "fakecatcher_cnn",
            "hybrid_ensemble",
        ],
        help="Experiment or pipeline step to run",
    )

    # Feature extraction
    parser.add_argument("--real_dir", help="Directory with real videos")
    parser.add_argument("--fake_dir", help="Directory with fake videos")
    parser.add_argument(
        "--output_csv",
        default="data/features.csv",
        help="Output CSV for extracted features",
    )

    # Random Forest ablation
    parser.add_argument(
        "--data",
        help="Path to features CSV for Random Forest ablation",
    )

    # CNN FakeCatcher
    parser.add_argument(
        "--maps_dir",
        help="Directory containing PPG map folders: spatial/ and spectral/",
    )
    parser.add_argument(
        "--models_dir",
        default="models",
        help="Directory to save trained CNN models",
    )

    # Hybrid ensemble
    parser.add_argument(
        "--csv",
        help="Feature CSV used by hybrid ensemble",
    )
    parser.add_argument(
        "--npz",
        help="Directory containing spectral PPG .npz files",
    )
    parser.add_argument(
        "--cnn",
        help="Path to trained CNN .keras model",
    )

    args = parser.parse_args()

    if args.exp == "extract_features":
        if not args.real_dir or not args.fake_dir:
            raise ValueError(
                "--real_dir and --fake_dir are required for extract_features"
            )

        extract_features.run(
            real_dir=args.real_dir,
            fake_dir=args.fake_dir,
            output_csv=args.output_csv,
        )

    elif args.exp == "ablation_rf":
        if not args.data:
            raise ValueError("--data is required for ablation_rf")

        ablation_rf.run(
            data_path=args.data,
        )

    elif args.exp == "fakecatcher_cnn":
        if not args.maps_dir:
            raise ValueError("--maps_dir is required for fakecatcher_cnn")

        fakecatcher_ppg_cnn.run(
            maps_dir=args.maps_dir,
            models_dir=args.models_dir,
        )

    elif args.exp == "hybrid_ensemble":
        if not args.csv or not args.npz or not args.cnn:
            raise ValueError(
                "--csv, --npz, and --cnn are required for hybrid_ensemble"
            )

        hybrid_ensemble.run(
            csv_path=args.csv,
            npz_folder=args.npz,
            cnn_path=args.cnn,
        )


if __name__ == "__main__":
    main()
