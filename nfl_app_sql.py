import _cffi_backend
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas_gbq as pdgbq
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb
import altair as alt
from PIL import Image

# Adjust page layout
st.set_page_config(layout="wide")

# Creates API client
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

# Memoize function executions -> optimize performance
@st.cache(hash_funcs={_cffi_backend.__CDataGCP: hash})
def getData(credentials):
    # Creates query to import nfl data from gcp sql database
    query = """
    SELECT * 
    FROM `nfl-prediction-344002.nfl_data.nfl_weekly_data`
    """
    input_df = pdgbq.read_gbq(query, credentials=credentials)

    query = """
    SELECT * 
    FROM `nfl-prediction-344002.nfl_data.aggregate_nfl_data`
    """
    raw_agg_df = pdgbq.read_gbq(query, credentials=credentials)
    
    return input_df, raw_agg_df

input_df, raw_agg_df = getData(credentials)

# Creates app title and description
st.title('NFL Football Stats Explorer')
st.markdown('')

# Sidebar - Year and Week
st.sidebar.header('User Inputs')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(2000, 2022))))
selected_week = st.sidebar.selectbox('Week', list(range(1, 22)))

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
# selected_stats = st.sidebar.multiselect('Features',stats,stats)

# Uses function to load and filter data
df = filterData(input_df, selected_week, selected_year)

# Filtering data
# filtered_df = df[df.columns.intersection(selected_stats)]
filtered_df = df
filtered_df.set_index('Team Name', inplace=True)

# Displaying dataframe in app
st.header('Weekly Statistics')
st.write('Dimensions: ' + str(filtered_df.shape[0]) + ' rows and ' + str(filtered_df.shape[1]) + ' columns')
st.dataframe(filtered_df)

agg_df = raw_agg_df.sort_values(by=['year', 'week'], ignore_index=True)

# Two columns for training and testing data
col1, col2 = st.columns(2)

if selected_week == 1:
    
    train_df = agg_df[agg_df.year < selected_year]
    
    temp_df = agg_df[agg_df.year == selected_year]
    test_df = temp_df[temp_df.week == selected_week]
    
else:
    
    temp_df2 = agg_df[agg_df.year == selected_year]
    temp_df2 = temp_df2[temp_df2.week < selected_week]
    
    ind = temp_df2.last_valid_index()
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
st.write("Top 5 Predictors")
top = resultCorr.abs().sort_values(by='result', axis=1, ascending=False)
top5 = resultCorr[top.columns[:5]]
st.dataframe(top5)

top5box = st.checkbox('Only train with top 5 predictors')

if not top5box:
    X_train = train_df.drop(columns=['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'year', 'result'])
    X_test = test_df.drop(columns=['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'year', 'result'])
else:
    X_train = train_df[top5.columns]
    X_test = test_df[top5.columns]
    
Y_train = train_df[['result']] 
Y_test = test_df[['result']]

clf = LogisticRegression(penalty='l1', dual=False, tol=0.001, C=1.0, fit_intercept=True, 
                   intercept_scaling=1, class_weight='balanced', random_state=None, 
                   solver='liblinear', max_iter=1000, multi_class='ovr', verbose=0)

clf.fit(X_train, np.ravel(Y_train.values))
y_pred = clf.predict_proba(X_test)

def display(row):
    chart_data = pd.DataFrame(
        row,
        index=["percentage"],
    )

    # Convert wide-form data to long-form
    data = pd.melt(chart_data.reset_index(), id_vars=["index"])

    # Horizontal stacked bar chart
    chart = (alt.Chart(data).mark_bar().encode(
            x=alt.X("value", type="quantitative", title="", axis=None),
            y=alt.Y("index", type="nominal", title="", axis=None),
            color=alt.Color("variable", type="nominal", title="", legend=None),
        ).configure_axis(grid=False).configure_view(strokeWidth=0))

    st.markdown("""
        <style type='text/css'>
            details {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)

    st.altair_chart(chart, use_container_width=True)

# Import images into dictionary
logos = {
    "den": "./images/broncos.png",
    "oti": "./images/titans.png",
    "rav": "./images/ravens.png",
    "sdg": "./images/chargers.png",
    "gnb": "./images/packers.png",
    "dal": "./images/cowboys.png",
    "nwe": "./images/patriots.png",
    "phi": "./images/eagles.png",
    "nyj": "./images/jets.png",
    "was": "./images/washington.png",
    "chi": "./images/bears.png",
    "tam": "./images/buccaneers.png",
    "kan": "./images/chiefs.png",
    "min": "./images/vikings.png",
    "car": "./images/panthers.png",
    "cin": "./images/bengals.png",
    "cle": "./images/browns.png",
    "ram": "./images/rams.png",
    "atl": "./images/falcons.png",
    "det": "./images/lions.png",
    "mia": "./images/dolphins.png",
    "nyg": "./images/giants.png",
    "htx": "./images/texans.png",
    "buf": "./images/bills.png",
    "sea": "./images/seahawks.png",
    "sfo": "./images/49ers.png",
    "rai": "./images/raiders.png",
    "nor": "./images/saints.png",
    "crd": "./images/cardinals.png",
    "jax": "./images/jaguars.png",
    "pit": "./images/steelers.png",
    "clt": "./images/colts.png"
    }

st.write("") 
st.header("Predictions")
only60 = st.checkbox("Only display predictions greater than 60%")

if only60:
    y_pred = y_pred[np.where((y_pred[:,1] >= 0.6) | (y_pred[:,1] <= 0.4))]
    
    j = 0
    for i in y_pred:

        teams_df = test_df.reset_index()
        away_team = teams_df.loc[j,'away_name']
        home_team = teams_df.loc[j,'home_name']
        
        away_abbr = teams_df.loc[j,'away_abbr']
        home_abbr = teams_df.loc[j,'home_abbr']
        
        j += 1
        
        col1, col2, col3 = st.columns([1,3,1])

        with col1:
            path = logos[home_abbr]
            image = Image.open(path)
            st.image(image, width=130, caption=home_team)

        with col2: 
            st.write("")
            st.write("")
            display(i.reshape((1,2)))
            
        with col3:
            path = logos[away_abbr]
            image = Image.open(path)
            st.image(image, width=130, caption=away_team)
            
else:
    j = 0
    for i in y_pred:

        teams_df = test_df.reset_index()
        away_team = teams_df.loc[j,'away_name']
        home_team = teams_df.loc[j,'home_name']
        
        away_abbr = teams_df.loc[j,'away_abbr']
        home_abbr = teams_df.loc[j,'home_abbr']
        
        j += 1
        
        col1, col2, col3 = st.columns([1,3,1])

        with col1:
            path = logos[home_abbr]
            image = Image.open(path)
            st.image(image, width=130, caption=home_team)

        with col2: 
            st.write("")
            st.write("")
            display(i.reshape((1,2)))
            
        with col3:
            path = logos[away_abbr]
            image = Image.open(path)
            st.image(image, width=130, caption=away_team)
       


