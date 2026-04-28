from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def make_pipeline(use_smote=False, random_state=42):
    steps = []

    steps.append(("scaler", StandardScaler()))

    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))

    clf = RandomForestClassifier(
        random_state=random_state,
        class_weight="balanced"
    )

    steps.append(("clf", clf))
    return Pipeline(steps)
