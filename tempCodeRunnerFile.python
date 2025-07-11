import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv('C:/Users/amrit/Desktop/prediction/student/student-mat.csv', delimiter=';')
X = df.drop('G3', axis=1)
y = df['G3'] >= 10  # Binary classification: Pass if G3 >= 10

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Apply one-hot encoding
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'student_model.pkl')