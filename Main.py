!pip install pandas scikit-learn transformers torch accelerate

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("heart.csv")  # Public data base from Kaggle CSV

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = 100 * accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy:.2f} %")


# comment
feature_importance = pd.Series(
    model.coef_[0],
    index=df.drop("target", axis=1).columns
).sort_values(key=abs, ascending=False)

feature_importance.head()
