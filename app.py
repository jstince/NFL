import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Adjust page layout
st.set_page_config(layout="wide")

# Creates app title and description
st.title('NFL Football Stats Explorer')
st.markdown('')

# Loads data from directory
input_df = pd.read_csv(r'C:\Users\Jared Stinson\Documents\Visual Studio Code Projects\NFL Predictions\NFL History\nfl_history_00_21.csv')

# Sidebar - Year and Week
st.sidebar.header('User Inputs')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(2000, 2022))))
selected_week = st.sidebar.selectbox('Week', list(range(1, 22)))

# Memoize function executions -> optimize performance
@st.cache()

# Function to filter data based on sidebar inputs
def filterData(raw_df, week, year):
    
    # Filters dataframe by year and week
    raw_df = raw_df.loc[raw_df['year'] == year]
    raw_df = raw_df.loc[raw_df['week'] == week]
    
    # Renaming columns so headers will wrap text in web app (more compressed table)
    raw_df = raw_df.rename(columns={'team_name': 'Team Name',
                                    'team_abbr': 'Team Abbreviation',
                                    'score': 'Score',
                                    'game_won': 'Game Won',
                                    'game_lost': 'Game Lost',
                                    'first_downs': 'First Downs',
                                    'fourth_down_attempts': '4th Down Attempts',
                                    'fourth_down_conversions': '4th Down Conversions',
                                    'fumbles': 'Fumbles',
                                    'fumbles_lost': 'Fumbles Lost',
                                    'interceptions': 'Interceptions',
                                    'net_pass_yards': 'Net Pass Yards',
                                    'pass_attempts': 'Pass Attempts',
                                    'pass_completions': 'Pass Completions',
                                    'pass_touchdowns': 'Pass Touchdowns',
                                    'pass_yards': 'Pass Yards',
                                    'penalties': 'Penalties',
                                    'points': 'Points',
                                    'rush_attempts': 'Rush Attempts',
                                    'rush_touchdowns': 'Rush Touchdowns',
                                    'rush_yards': 'Rush Yards',
                                    'third_down_attempts': '3rd Down Attempts',
                                    'third_down_conversions': '3rd Down Conversions',
                                    'time_of_possession': 'Possession Time',
                                    'times_sacked': 'Times Sacked',
                                    'total_yards': 'Total Yards',
                                    'turnovers': 'Turnovers',
                                    'yards_from_penalties': 'Yards from Penalties',
                                    'yards_lost_from_sacks': 'Yards Lost from Sacks',
                                    'week': 'Week',
                                    'year': 'Year'})

    # raw_df.set_index('Team Name', inplace=True)

    # Dropping irrelevant columns
    raw_df = raw_df.drop(columns=['Points', 'Team Abbreviation', 'Week', 'Year'])

    return raw_df

# Sidebar - Statistics
stats = ['Team Name', 'Score', 'Game Won', 'Game Lost', 'First Downs', '4th Down Attempts',
        '4th Down Conversions', 'Fumbles', 'Fumbles Lost', 'Interceptions', 'Net Pass Yards', 'Pass Attempts',
        'Pass Completions', 'Pass Touchdowns', 'Pass Yards', 'Penalties', 'Rush Attempts', 'Rush Touchdowns',
        'Rush Yards', 'Third Down Attempts', 'Third Down Conversions', 'Possession Time', 'Times Sacked', 'Total Yards',
        'Turnovers', 'Yards From Penalties', 'Yards Lost From Sacks', 'Week', 'Year']
selected_stats = st.sidebar.multiselect('Statistics',stats,stats)

# Uses function to load and filter data
df = filterData(input_df, selected_week, selected_year)

# Filtering data
filtered_df = df[df.columns.intersection(selected_stats)]
filtered_df.set_index('Team Name', inplace=True)

# Displaying dataframe in app
st.header('Weekly Stats')
st.write('Dimensions: ' + str(filtered_df.shape[0]) + ' rows and ' + str(filtered_df.shape[1]) + ' columns')
st.dataframe(filtered_df)

# Two columns for training and testing data
col1, col2 = st.columns(2)

# Display aggregated data in app
agg_df = pd.read_csv(r"C:\Users\Jared Stinson\Documents\Visual Studio Code Projects\NFL Predictions\NFL History\agg_2000_2021.csv")

if selected_week == 1:
    prev_year = selected_year - 1
    temp_df = agg_df[agg_df.year == prev_year]
    
    if prev_year < 2021:
        temp_df = temp_df[temp_df.week == 21]
else:
    temp_df = agg_df[agg_df.year == selected_year]

ind = temp_df.last_valid_index()
train_df = agg_df[agg_df.index <= ind]

test_df = agg_df[agg_df.year == selected_year]
test_df = test_df[test_df.week == selected_week]

st.header("")
with col1:
    st.header("Training Data")
    st.write('Dimensions: ' + str(train_df.shape[0]) + ' rows and ' + str(train_df.shape[1]) + ' columns')
    st.dataframe(train_df)

with col2:
    st.header("Test Games")
    test_games_df = test_df[['away_name', 'home_name', 'week', 'year']]
    st.write('Dimensions: ' + str(test_games_df.shape[0]) + ' rows and ' + str(test_games_df.shape[1]) + ' columns')
    st.dataframe(test_games_df)

resultCorr = train_df.corr().tail(1).drop(columns=['week', 'year', 'result'])

# Display top 5 most useful predictors
st.write("")
st.write("Top 5 Predictors")
top = resultCorr.abs().sort_values(by='result', axis=1, ascending=False)
top5 = resultCorr[top.columns[:5]]
st.dataframe(top5)
# st.dataframe(resultCorr)

top5box = st.checkbox('Only train with top 5 predictors?')

if not top5box:
    X_train = train_df.drop(columns=['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'year', 'result'])
    X_test = test_df.drop(columns=['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'year', 'result'])
else:
    X_train = train_df[top5.columns]
    X_test = test_df[top5.columns]
    
Y_train = train_df[['result']] 
Y_test = test_df[['result']]

# X_train.to_csv('X_train.csv', index=False)
# Y_train.to_csv('Y_train.csv', index=False)
# X_test.to_csv('X_test.csv', index=False)
# Y_test.to_csv('Y_test.csv', index=False)

clf = LogisticRegression(penalty='l1', dual=False, tol=0.001, C=1.0, fit_intercept=True, 
                   intercept_scaling=1, class_weight='balanced', random_state=None, 
                   solver='liblinear', max_iter=1000, multi_class='ovr', verbose=0)

clf.fit(X_train, np.ravel(Y_train.values))
y_pred = clf.predict_proba(X_test)
y_pred = y_pred[:, 1]

only60 = st.checkbox("Only verify percentages greater than 60%?")
def display(y_pred, X_test):
    for g in range(len(y_pred)):
        win_prob = round(y_pred[g], 2) * 100
        win_prob = int(win_prob)
        
        away_team_df = test_df['away_name'].reset_index()
        home_team_df = test_df['home_name'].reset_index()
        
        away_team = away_team_df.loc[g, 'away_name']
        home_team = home_team_df.loc[g, 'home_name']
        
        st.write(f'The {away_team} had a {win_prob}% of beating the {home_team}.')
    
if not only60:
    display(y_pred, X_test)
else:
    y_pred2 = y_pred[y_pred>=0.6]
    y_pred3 = y_pred[y_pred<=0.4]
    
    predictions = np.concatenate((y_pred2, y_pred3))
    st.write(predictions)

# st.header("Logistic Regression Accuracy: " + str(100*accuracy_score(Y_test, np.round(y_pred)))+"%")


# dtrain = xgb.DMatrix(X_train, Y_train, feature_names=X_train.columns)
# dtest = xgb.DMatrix(X_test, Y_test, feature_names=X_test.columns)

# param = {'verbosity': 1, 
#          'objective':'binary:hinge',
#          'feature_selector': 'shuffle',
#          'booster':'gblinear',
#          'eval_metric' : 'error',
#          'learning_rate': 0.05}

# evallist = [(dtrain, 'train'), (dtest, 'test')]

# num_round = 300
# bst = xgb.train(param, dtrain, num_round)

# boost_pred = bst.predict(dtest)

# st.header("Boosting Model Accuracy: " + str(100*accuracy_score(Y_test, boost_pred)) + "%")
