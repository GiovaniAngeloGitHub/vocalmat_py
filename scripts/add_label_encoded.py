import pandas as pd

# Read the CSV file
df = pd.read_csv('data/annotations.csv')

# Add label_encoded column (0 for noise, 1 for vocalization)
df['label_encoded'] = (df['label'] == 'vocalization').astype(int)

# Save the updated CSV
df.to_csv('data/annotations.csv', index=False) 