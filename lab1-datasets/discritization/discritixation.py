import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv('salesdata.csv')
print("Initial Data:\n", df.head())

# Step 2: Define bins and labels for discretization
# Adjust bin edges based on distribution
bins = [0, 10000, 25000, float('inf')]
labels = ['Low', 'Medium', 'High']

# Step 3: Apply discretization
df['SalesCategory'] = pd.cut(df['Sales'], bins=bins, labels=labels, right=True)

print("\nData after Discretization:\n", df)

# Step 4: Analyze the distribution
sales_category_distribution = df['SalesCategory'].value_counts().sort_index()
print("\nSales Category Distribution:\n", sales_category_distribution)

# Optional: Save to a new CSV file
df.to_csv('discretized_sales_data.csv', index=False)
