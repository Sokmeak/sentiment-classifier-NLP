from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class Models:
    @staticmethod
    def logistic():
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42,solver="lbfgs")
        

    @staticmethod
    def random_forest():
        return RandomForestClassifier(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42
        )
