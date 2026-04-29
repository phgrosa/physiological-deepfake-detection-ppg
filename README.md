# physiological-deepfake-detection-ppg
Deepfake video detection using physiological signals (PPG maps) and interpretable handcrafted features.

This project implements multiple approaches for deepfake detection using:

- Physiological signals (PPG-based FakeCatcher)
- Motion and texture features (Random Forest)
- Hybrid ensemble combining CNN + RF with bias correction
- SHAP-based explainability

Dataset: CelebDF / FaceForensics++

## 🔬 Contributions

- Implementation of FakeCatcher-inspired CNN
- Motion + physiological feature extraction pipeline
- Bias-corrected hybrid ensemble improving F1 score
- Robust SHAP analysis for model interpretability
