import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

season_stats = pd.read_csv('josh_allen_24.csv',header=[0,1]) #load dataset with multilevel header
#print(season_stats.columns) #look at the columns
season_stats = season_stats.iloc[:-1]  # Exclude the last row
season_stats['Result','Result'] = season_stats['Result','Result'].str[0] #get only W or L
# Replace 'W', 'L', 'T' with numeric values
season_stats['Result','Result'] = season_stats['Result','Result'].replace({'W': 1, 'L': 0, 'T': 0})


season_stats.reset_index(drop=True, inplace=True)
season_stats.columns = [' '.join(col).strip() for col in season_stats.columns]

# Check the Result value counts
print(season_stats['Result Result'].value_counts())

# Select columns for analysis and include the Result column
stats = ['Passing Yds', 'Passing Cmp%', 'Passing TD', 'Passing Int', 'Result Result', 'Rushing Yds', 'Rushing TD' ]

# Filter the DataFrame to only include relevant columns
pairplot_data = season_stats[stats]

# Rename the 'Result Result' column for simplicity
pairplot_data = pairplot_data.rename(columns={'Result Result': 'Result'})

# Pairplot with hue for Result
sns.pairplot(pairplot_data, hue='Result', diag_kind='kde', palette={0: "red", 1: "green"})
plt.show()

# Features and target
X = season_stats[['Passing Yds', 'Passing Cmp%', 'Passing TD', 'Passing Int']]
y = season_stats['Result Result']

X.dropna(inplace=True)
y = y[X.index]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# create the classifier
lrc = LogisticRegression()

# fit classifier to the training data
lrc.fit(X_train,y_train)

y_pred = lrc.predict(X_test)
y_prob = lrc.predict_proba(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

#See which feature has the biggest effect
# Calculate feature importance
feature_importance = pd.Series(lrc.coef_[0], index=X.columns)

# Find the most important feature (by absolute value of the coefficient)
most_important_feature = feature_importance.abs().idxmax()
most_important_value = feature_importance[most_important_feature]

# Print all feature importances
print("Feature Importance:")
print(feature_importance)

win_yards = season_stats[season_stats['Result Result'] == 1]['Passing Yds']
loss_yards = season_stats[season_stats['Result Result'] == 0]['Passing Yds']
print(f"Average Passing Yards in Wins and number of wins: {win_yards.mean():.2f}, {win_yards.count()}")
print(f"Average Passing Yards in Losses and number of losses: {loss_yards.mean():.2f}, {loss_yards.count()}")


# Highlight the most important feature
print(f"\n'{most_important_feature}' with a coefficient of {most_important_value:.4f} "
      "is the most important statistic for deciding wins.")

X_test_original = scaler.inverse_transform(X_test)
yard_threshold = 200
over_under_predictions = (X_test_original[:, 0] > yard_threshold).astype(int)
for i, prob in enumerate(y_prob):
    print(f"Future Game number {i+1}: Predicted Win: {y_pred[i]}, Win Probability: {prob[1]:.2f}, Expectedd Passing Yds: {X_test_original[i, 0]:.2f}, Over  {yard_threshold} Yds: {over_under_predictions[i]}")

#Lets move to analyzing his career trajectory
total_yards = 18055
total_tds = 195	
games_played = 111

yards_per_game = total_yards / games_played
tds_per_game = total_tds / games_played

# Assumptions
games_per_season = 17
remaining_seasons = 10  # Assumes 10 more seasons
decline_factor = 0.95  # 5% decline each season after 5 seasons

# Projection
projected_yards = []
projected_tds = []
cumulative_yards = total_yards
cumulative_tds = total_tds

for season in range(1, remaining_seasons + 1):
    if season > 5:  # Apply decline after 5 seasons
        yards_per_game *= decline_factor
        tds_per_game *= decline_factor
    
    season_yards = yards_per_game * games_per_season
    season_tds = tds_per_game * games_per_season

    cumulative_yards += season_yards
    cumulative_tds += season_tds  

    # Append cumulative stats to the projections
    projected_yards.append(cumulative_yards)
    projected_tds.append(cumulative_tds)


# Total Career Projections
total_projected_yards = sum(projected_yards)
total_projected_tds = sum(projected_tds)

#Print the results
print(f"Projected Career Passing Yards: {total_projected_yards:.0f}")
print(f"Projected Career Passing Touchdowns: {total_projected_tds:.0f}")




