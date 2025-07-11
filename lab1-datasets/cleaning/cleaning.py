import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv('employee.csv')
print("Initial Data:\n", df.head())

# Step 2: Handle missing values
# Fill missing 'Age' and 'Salary' with their respective mean values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].mean(), inplace=True)

# Convert 'Age' and 'Salary' to integer type
df['Age'] = df['Age'].astype(int)
df['Salary'] = df['Salary'].astype(int)

# Step 3: Standardize department names
df['Department'] = df['Department'].replace({
    'Human Resources': 'HR',
    'H.R.': 'HR',
    'hr': 'HR'
})

# Step 4: Remove duplicate records based on 'ID'
df.drop_duplicates(subset='ID', keep='first', inplace=True)

# Step 5: Show cleaned data
print("\nCleaned Data:\n", df)

# Optional: Save cleaned data to a new CSV
df.to_csv('cleaned_employee.csv', index=False)
