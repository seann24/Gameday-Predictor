import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

season_stats = pd.read_csv('josh_allen_24.csv',header=[0,1]) #load dataset with multilevel header
print(season_stats.columns) #look at the columns
season_stats = season_stats.iloc[:-1]  # Exclude the last row
season_stats['Result','Result'] = season_stats['Result','Result'].str[0] #get only W or L
# Replace 'W', 'L', 'T' with numeric values
season_stats['Result','Result'] = season_stats['Result','Result'].replace({'W': 1, 'L': 0, 'T': 0})


season_stats.reset_index(drop=True, inplace=True)
# check result value counts
print(season_stats[('Result', 'Result')].value_counts())

plt.figure(figsize=(8, 6))
sns.boxplot(
    x=season_stats[('Result', 'Result')],
    y=season_stats[('Passing', 'Yds')]
)

# Customize labels
plt.xticks(ticks=[0, 1], labels=['Loss/Tie', 'Win'])
plt.title("Josh Allen's Passing Yards vs. Game Result")
plt.xlabel("Game Result")
plt.ylabel("Passing Yards")
plt.show()

