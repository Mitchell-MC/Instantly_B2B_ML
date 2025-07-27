import pandas as pd
df = pd.read_csv('merged_contacts.csv') 
print('Shape:', df.shape) 
print('Columns:', len(df.columns))
print('Memory usage:', df.memory_usage(deep=True).sum() / 1024**2, 'MB')
print('Missing values:', df.isnull().sum().sum()); print('Sample columns:', list(df.columns[:10]))