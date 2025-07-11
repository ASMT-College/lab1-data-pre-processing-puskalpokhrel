import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv('customer_ages.csv')
print("Initial Data:\n", df.head())

# Step 2: Create bins and assign labels
# Bins: [18-29], [30-49], [50-100] -> with right=False so upper bound is excluded
bins = [18, 31, 51, 101]  # right=False means [start, end)
labels = ['Young', 'Middle-aged', 'Senior']

# Assign bins using pd.cut
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

print("\nData after Binning:\n", df)

# Step 3: Calculate the distribution of customers in each age group
age_group_distribution = df['AgeGroup'].value_counts().sort_index()
print("\nAge Group Distribution:\n", age_group_distribution)

# Optional: Save the binned dataset
df.to_csv('binned_customer_ages.csv', index=False)
