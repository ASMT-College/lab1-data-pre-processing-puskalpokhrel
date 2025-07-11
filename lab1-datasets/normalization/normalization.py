import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the dataset
df = pd.read_csv('student_scores.csv')
print("Initial Data:\n", df.head())

# Step 2: Preserve original scores for side-by-side comparison
original_df = df.copy()

# Step 3: Apply Min-Max normalization (0 to 1 scale)
scaler = MinMaxScaler()
normalized_scores = scaler.fit_transform(df[['Math', 'Science', 'English']])

# Step 4: Create DataFrame for normalized scores
normalized_df = pd.DataFrame(normalized_scores, columns=['Math_Norm', 'Science_Norm', 'English_Norm'])

# Step 5: Concatenate original and normalized data
final_df = pd.concat([original_df, normalized_df], axis=1)

# Step 6: Display the result
print("\nOriginal and Normalized Scores:\n", final_df)

# Optional: Save to CSV
final_df.to_csv('normalized_student_scores.csv', index=False)
