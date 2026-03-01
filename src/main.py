
# ------------------------------
# Student Performance Analyzer
# ------------------------------

# Step 1: Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 2: Load CSV
script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, '../data/student-mat.csv')

df = pd.read_csv(csv_path, sep=';')
print("CSV loaded successfully!\n")

# Step 3: Clean / prepare data
numeric_cols = ['age','Medu','Fedu','traveltime','studytime','failures',
                'famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Step 4: Explore
print("First 5 rows:\n", df.head(), "\n")
print("Data Info:\n")
print(df.info(), "\n")
print("Summary statistics:\n")
print(df.describe(), "\n")

# Step 5: Visualizations
plt.figure(figsize=(8,5))
sns.histplot(df['g3'], bins=10, kde=True)
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("G3")
plt.ylabel("Number of Students")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='sex', y='g3', data=df)
plt.title("Final Grades by Gender")
plt.show()

numeric_df = df.select_dtypes(include='number')
plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (numeric columns only)")
plt.show()

# Step 6: Predictive Model
features = ['g1','g2','studytime','failures','absences']
X = df[features]
y = df['g3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
print(f"Model trained! Accuracy on test set: {model.score(X_test, y_test):.2f}\n")

# Step 7: Predict new student grades in a loop
predictions = []

print("Enter new student data to predict G3 (type 'exit' to stop):\n")

while True:
    g1_input = input("G1 (first term grade, 0-20): ")
    if g1_input.lower() == 'exit':
        break
    g2_input = input("G2 (second term grade, 0-20): ")
    if g2_input.lower() == 'exit':
        break
    studytime_input = input("Study time (1-4): ")
    if studytime_input.lower() == 'exit':
        break
    failures_input = input("Past failures: ")
    if failures_input.lower() == 'exit':
        break
    absences_input = input("Absences: ")
    if absences_input.lower() == 'exit':
        break

    # Convert to integers
    g1 = int(g1_input)
    g2 = int(g2_input)
    studytime = int(studytime_input)
    failures = int(failures_input)
    absences = int(absences_input)

    predicted_G3 = model.predict([[g1, g2, studytime, failures, absences]])[0]
    print(f"Predicted final grade (G3): {predicted_G3:.2f}\n")
    
    predictions.append({
        'G1': g1,
        'G2': g2,
        'Studytime': studytime,
        'Failures': failures,
        'Absences': absences,
        'Predicted_G3': round(predicted_G3,2)
    })

# Print summary of all predictions
if predictions:
    print("\nSummary of all predictions:")
    summary_df = pd.DataFrame(predictions)
    print(summary_df)
import pandas as pd

print("Student Performance Analyzer Started !!!!")
