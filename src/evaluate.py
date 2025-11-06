import joblib, json, os
from sklearn.metrics import accuracy_score, classification_report
from src.train import load_and_preprocess

def evaluate(model_path='model/churn_model.pkl'):
    df = load_and_preprocess()
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    # split sencillo
    X_train = X.sample(frac=0.8, random_state=42)
    X_test = X.drop(X_train.index)
    y_train = y.loc[X_train.index]
    y_test = y.drop(y_train.index)
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    out = {'accuracy': accuracy_score(y_test, y_pred), 'report': classification_report(y_test, y_pred, output_dict=True)}
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/eval_report.json','w') as f:
        json.dump(out, f, indent=2)
    print("Evaluación guardada en artifacts/eval_report.json")

if __name__ == "__main__":
    evaluate()
