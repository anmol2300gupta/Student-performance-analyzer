# ------------------------------
# Batch Prediction for Student Grades
# ------------------------------

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1: Load original dataset
script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, '../data/student-mat.csv')
df = pd.read_csv(csv_path, sep=';')

# Step 2: Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Step 3: Convert numeric columns
numeric_cols = ['age','medu','fedu','traveltime','studytime','failures',
                'famrel','freetime','goout','dalc','walc','health','absences','g1','g2','g3']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

# Step 4: Select features and target
features = ['g1','g2','studytime','failures','absences']  # simple numeric features
X = df[features]
y = df['g3']

# Step 5: Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print(f"Model trained! Test set R² score: {model.score(X_test, y_test):.2f}")

# Step 6: Visualize predicted vs actual G3
y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Predicted vs Actual G3 - Test Set")
plt.show()

# Step 7: Load new students CSV for batch prediction
new_csv_path = os.path.join(script_dir, '../data/new_students.csv')
new_students = pd.read_csv(new_csv_path)

# Step 8: Predict G3 for new students
new_students['predicted_G3'] = model.predict(new_students[features])

# Step 9: Save predictions
output_path = os.path.join(script_dir, '../data/predicted_G3.csv')
new_students.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")
print(new_students)