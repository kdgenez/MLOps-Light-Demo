import pandas as pd
import joblib, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_and_preprocess(path='data/Telco-Customer-Churn.csv'):
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df['Churn'] = df['Churn'].map({'Yes':1,'No':0})
    df = df.drop(columns=['customerID'])
    df = pd.get_dummies(df, drop_first=True)
    return df

def train(output='model/churn_model.pkl'):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df = load_and_preprocess()
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    joblib.dump(model, output)
    print("Modelo guardado en", output)

if __name__ == "__main__":
    train()
