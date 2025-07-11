import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the dataset
df = pd.read_csv('medicaldata.csv')
print("Initial Data:\n", df.head())

# Step 2: Define features and target variable
X = df.drop(columns=['PatientID', 'Disease'])  # Remove PatientID (non-informative)
y = df['Disease']

# Step 3: Normalize features (Chi-square requires non-negative data)
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Step 4: Apply Chi-square feature selection
selector = SelectKBest(score_func=chi2, k=3)
selector.fit(X_scaled, y)

# Step 5: Get top 3 features
top_features = X.columns[selector.get_support()]
print("\nTop 3 Features for Predicting Disease:\n", top_features)

# Step 6: Optional - Show scores for all features
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Chi2 Score': selector.scores_
}).sort_values(by='Chi2 Score', ascending=False)

print("\nAll Feature Scores:\n", feature_scores)
