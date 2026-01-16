from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

class Evaluator:
    @staticmethod
    def evaluate(model, X, y, name, file_path=None):
        pred = model.predict(X)

        acc = accuracy_score(y, pred)
        cm = confusion_matrix(y, pred)
        report = classification_report(y, pred)

        output = []
        output.append("=" * 60)
        output.append(name)
        output.append(f"Accuracy: {acc}")
        output.append("Confusion Matrix:")
        output.append(str(cm))
        output.append("Classification Report:")
        output.append(report)

        result_text = "\n".join(output)

        # Print to console
        print(result_text)

        # Write to file if path provided
        if file_path:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(result_text + "\n\n")
