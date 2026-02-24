import pandas as pd
import glob
print("Counting Age Cohorts in all Top 5 League data")
files = glob.glob('data/*_merged.csv') + glob.glob('data/*Advanced Stats.csv')
dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        if 'Age' in df.columns:
            dfs.append(df)
    except Exception:
        pass
if dfs:
    df_all = pd.concat(dfs)
    if 'Age' in df_all.columns:
        counts = df_all['Age'].value_counts().sort_index()
        for age, count in counts.items():
            print(f"Age {age}: {count} players")
else:
    print("No Age data found")
