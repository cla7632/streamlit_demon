import pandas as pd
import streamlit as st
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np



# Function to determine the color of the text based on value comparison
def get_color(current, previous):
    if current > previous:
        return "green"
    elif current < previous:
        return "red"
    else:
        return "white"

# Function to get the percentage change and arrow
def get_change_and_arrow(current, previous):
    if previous == 0:
        return "", ""
    change = ((current - previous) / previous) * 100
    if abs(change) < 0.01:
        change_text = f" (<0.01%) →"
    else:
        if change == 0:
            change_text = f" ({change:.2f}%) →"
        else:
            arrow = "↑" if change > 0 else "↓"
            change_text = f" ({change:.2f}%) {arrow}"
    return change_text

# Custom CSS to change the background color
st.markdown(
    """
    <style>
    .main {
        background-color: #001871;
    }
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(255, 255, 0, 1);
        background-color: #0E1117;
        margin-bottom: 10px;
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    .metric-card div {
        font-size: 20px;
    }
    .metric-card .value {
        font-size: 30px;
        font-weight: bold;
    }
    .metric-card .change {
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Data Preparation
data = {
    "quarter": ["1Q2022", "2Q2022", "3Q2022", "4Q2022", "1Q2023", "2Q2023", "3Q2023", "4Q2023"],
    "postpaidrevenue_rmm": [1270, 1272, 1287, 1290, 1283, 1267, 1257, 1268],
    "prepaidrevenue_rmm": [1118, 1145, 1125, 1148, 1137, 1144, 1146, 1146],
    "wholesaleothersrevenue_rmm": [258, 266, 261, 244, 243, 255, 265, 276],
    "homefibrerevenue_rmm": [31, 33, 35, 37, 40, 42, 45, 47],
    "postpaidmobile_000": [6573, 6613, 6652, 6672, 6726, 6772, 6864, 6938],
    "prepaid_000": [13039, 13174, 13216, 13313, 13459, 13592, 13614, 13483],
    "fibre_000": [86, 91, 96, 101, 107, 113, 121, 131],
    "totalsubscribers_000": [19698, 19878, 19964, 20086, 20291, 20478, 20600, 20552],
    "netaddpostpaid_000": [None, 40, 39, 20, 54, 46, 92, 74],
    "netaddprepaid_000": [None, 135, 42, 96, 146, 134, 22, -132],
    "netaddfibre_000": [None, 5, 5, 6, 6, 6, 8, 10],
    "totalnetaddsubscribers_000": [None, 180, 86, 122, 205, 186, 122, -48],
    "postpaidmobilearpu_rm": [71, 70, 71, 70, 69, 68, 67, 66],
    "prepaidarpu_rm": [29, 29, 29, 29, 28, 28, 28, 28],
    "blendedmobilearpu_rm": [42, 42, 42, 42, 41, 41, 40, 40],
    "fibrearpu_rm": [122, 124, 124, 124, 126, 127, 126, 124],
    "blendedarpu_rm": [42, 42, 42, 42, 42, 41, 40, 41]
}

df = pd.DataFrame(data)

# Fill missing values
df.fillna(0, inplace=True)

# Calculate growth rates
df['q_qgrowth'] = df['postpaidrevenue_rmm'].pct_change() * 100
df['y_ygrowth'] = df['postpaidrevenue_rmm'].pct_change(periods=4) * 100

# Function to forecast ARPU
def forecast_arpu(data, periods=1):
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    return forecast

# Predicting ARPU for the next quarter
postpaid_arpu_forecast = forecast_arpu(df['postpaidmobilearpu_rm'], periods=1)
prepaid_arpu_forecast = forecast_arpu(df['prepaidarpu_rm'], periods=1)
blended_arpu_forecast = forecast_arpu(df['blendedmobilearpu_rm'], periods=1)
fibre_arpu_forecast = forecast_arpu(df['fibrearpu_rm'], periods=1)


# Calculate percentage change compared to the latest quarter
latest_postpaid_arpu = df['postpaidmobilearpu_rm'].iloc[-1]
latest_prepaid_arpu = df['prepaidarpu_rm'].iloc[-1]
latest_blended_arpu = df['blendedmobilearpu_rm'].iloc[-1]
latest_fibre_arpu = df['fibrearpu_rm'].iloc[-1]

postpaid_arpu_change = ((postpaid_arpu_forecast.iloc[0] - latest_postpaid_arpu) / latest_postpaid_arpu) * 100
prepaid_arpu_change = ((prepaid_arpu_forecast.iloc[0] - latest_prepaid_arpu) / latest_prepaid_arpu) * 100
blended_arpu_change = ((blended_arpu_forecast.iloc[0] - latest_blended_arpu) / latest_blended_arpu) * 100
fibre_arpu_change = ((fibre_arpu_forecast.iloc[0] - latest_fibre_arpu) / latest_fibre_arpu) * 100

# Creating a synthetic target variable for churn
df['churn'] = (df['totalnetaddsubscribers_000'] < 0).astype(int)

# Feature selection and splitting data
features = ['postpaidrevenue_rmm', 'prepaidrevenue_rmm', 'wholesaleothersrevenue_rmm', 
            'homefibrerevenue_rmm', 'postpaidmobile_000', 'prepaid_000', 'fibre_000', 
            'postpaidmobilearpu_rm', 'prepaidarpu_rm', 'blendedmobilearpu_rm', 'fibrearpu_rm']
X = df[features]
y = df['churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict churn for the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Churn Prediction Accuracy: {accuracy:.2f}')

# Predict churn for the next quarter
next_quarter_features = np.array([[
    df['postpaidrevenue_rmm'].iloc[-1],
    df['prepaidrevenue_rmm'].iloc[-1],
    df['wholesaleothersrevenue_rmm'].iloc[-1],
    df['homefibrerevenue_rmm'].iloc[-1],
    df['postpaidmobile_000'].iloc[-1],
    df['prepaid_000'].iloc[-1],
    df['fibre_000'].iloc[-1],
    postpaid_arpu_forecast.iloc[0],
    prepaid_arpu_forecast.iloc[0],
    blended_arpu_forecast.iloc[0],
    fibre_arpu_forecast.iloc[0]
]])

churn_forecast = model.predict(next_quarter_features)
churn_probability = model.predict_proba(next_quarter_features)[0][1]

print("Churn Forecast for Next Quarter:", churn_forecast[0])
print("Churn Probability for Next Quarter:", churn_probability)

# Streamlit Dashboard
st.markdown('<h1 style="color: yellow;">Telco ARPU and Subscriber Analytics & Prediction</h1>', unsafe_allow_html=True)

# Metric Cards for Latest Quarter Information
latest_quarter = df.iloc[-1]
previous_quarter = df.iloc[-2]


st.subheader('Latest Quarter (Q4 2023) Information')

# Function to determine the color of the text based on value comparison
def get_color(current, previous):
    if current > previous:
        return "green"
    elif current < previous:
        return "red"
    else:
        return "white"

# Function to get the arrow based on value comparison
def get_arrow(current, previous):
    if current > previous:
        return "↑"
    elif current < previous:
        return "↓"
    else:
        return "→"

# Create Metric Cards
subscriber_metrics = {
    "Total Subscribers (000)": (latest_quarter['totalsubscribers_000'], previous_quarter['totalsubscribers_000']),
    "Total Postpaid Subscribers (000)": (latest_quarter['postpaidmobile_000'], previous_quarter['postpaidmobile_000']),
    "Total Prepaid Subscribers (000)": (latest_quarter['prepaid_000'], previous_quarter['prepaid_000']),
    "Total Fibre Subscribers (000)": (latest_quarter['fibre_000'], previous_quarter['fibre_000'])
}

net_adds_metrics = {
    "Net Adds Postpaid (000)": (latest_quarter['netaddpostpaid_000'], previous_quarter['netaddpostpaid_000']),
    "Net Adds Prepaid (000)": (latest_quarter['netaddprepaid_000'], previous_quarter['netaddprepaid_000']),
    "Net Adds Fibre (000)": (latest_quarter['netaddfibre_000'], previous_quarter['netaddfibre_000'])
}

arpu_metrics = {
    "Postpaid ARPU (RM)": (latest_quarter['postpaidmobilearpu_rm'], previous_quarter['postpaidmobilearpu_rm']),
    "Prepaid ARPU (RM)": (latest_quarter['prepaidarpu_rm'], previous_quarter['prepaidarpu_rm']),
    "Blended ARPU (RM)": (latest_quarter['blendedmobilearpu_rm'], previous_quarter['blendedmobilearpu_rm']),
    "Fibre ARPU (RM)": (latest_quarter['fibrearpu_rm'], previous_quarter['fibrearpu_rm'])
}

# Create Metric Cards
revenue_metrics = {
    "Postpaid Revenue (RM Million)": (latest_quarter['postpaidrevenue_rmm'], previous_quarter['postpaidrevenue_rmm']),
    "Prepaid Revenue (RM Million)": (latest_quarter['prepaidrevenue_rmm'], previous_quarter['prepaidrevenue_rmm']),
    "Wholesale/Others Revenue (RM Million)": (latest_quarter['wholesaleothersrevenue_rmm'], previous_quarter['wholesaleothersrevenue_rmm']),
    "Home Fibre Revenue (RM Million)": (latest_quarter['homefibrerevenue_rmm'], previous_quarter['homefibrerevenue_rmm'])
}

# Function to display metric cards
def display_metric_cards(metrics):
    cols = st.columns(len(metrics))
    for i, (metric, (current, previous)) in enumerate(metrics.items()):
        color = get_color(current, previous)
        arrow = get_arrow(current, previous)
        percentage_change = ((current - previous) / previous) * 100 if previous != 0 else 0
        cols[i].markdown(
            f"""
            <div class="metric-card">
                <div>{metric}</div>
                <div class="value" style="color: {color};">{current} {arrow}</div>
                <div class="change" style="color: {color};">Change: {percentage_change:.2f}%</div>
            </div>
            """, unsafe_allow_html=True
        )



st.subheader("Subscriber Metrics")
display_metric_cards(subscriber_metrics)

st.subheader("Net Adds Metrics")
display_metric_cards(net_adds_metrics)

st.subheader("ARPU Metrics")
display_metric_cards(arpu_metrics)

st.subheader("Revenue Metrics")
display_metric_cards(revenue_metrics)

# ARPU Forecast
st.subheader('ARPU Forecast for Next Quarter')

# Predicting ARPU for the next quarter
postpaid_arpu_forecast = forecast_arpu(df['postpaidmobilearpu_rm'], periods=1)
prepaid_arpu_forecast = forecast_arpu(df['prepaidarpu_rm'], periods=1)
blended_arpu_forecast = forecast_arpu(df['blendedmobilearpu_rm'], periods=1)
fibre_arpu_forecast = forecast_arpu(df['fibrearpu_rm'], periods=1)

# Round forecasted ARPU values to 2 decimal points
postpaid_arpu_forecast_value_rounded = round(postpaid_arpu_forecast.iloc[0], 2)
prepaid_arpu_forecast_value_rounded = round(prepaid_arpu_forecast.iloc[0], 2)
blended_arpu_forecast_value_rounded = round(blended_arpu_forecast.iloc[0], 2)
fibre_arpu_forecast_value_rounded = round(fibre_arpu_forecast.iloc[0], 2)

# Calculate percentage change compared to the latest quarter
latest_postpaid_arpu = df['postpaidmobilearpu_rm'].iloc[-1]
latest_prepaid_arpu = df['prepaidarpu_rm'].iloc[-1]
latest_blended_arpu = df['blendedmobilearpu_rm'].iloc[-1]

latest_fibre_arpu = df['fibrearpu_rm'].iloc[-1]

# Calculate percentage change compared to the latest quarter for ARPU
postpaid_arpu_change = ((postpaid_arpu_forecast_value_rounded - latest_postpaid_arpu) / latest_postpaid_arpu) * 100
prepaid_arpu_change = ((prepaid_arpu_forecast_value_rounded - latest_prepaid_arpu) / latest_prepaid_arpu) * 100
blended_arpu_change = ((blended_arpu_forecast_value_rounded - latest_blended_arpu) / latest_blended_arpu) * 100
fibre_arpu_change = ((fibre_arpu_forecast_value_rounded - latest_fibre_arpu) / latest_fibre_arpu) * 100

# Function to determine the color and arrow based on percentage change
def get_color_and_arrow(change):
    if change > 0:
        return "green", "↑"
    elif change < 0:
        return "red", "↓"
    else:
        return "white", "→"

# Get color and arrow for each ARPU change
postpaid_color, postpaid_arrow = get_color_and_arrow(postpaid_arpu_change)
prepaid_color, prepaid_arrow = get_color_and_arrow(prepaid_arpu_change)
blended_color, blended_arrow = get_color_and_arrow(blended_arpu_change)
fibre_color, fibre_arrow = get_color_and_arrow(fibre_arpu_change)

# Display ARPU forecast metrics
st.markdown(
    f"""
    <div class="metric-card">
        <div>Postpaid ARPU Forecast</div>
        <div class="value" style="color: {postpaid_color};">RM{postpaid_arpu_forecast_value_rounded}  {postpaid_arrow}</div>
        <div class="change" style="color: {postpaid_color};">Change: {postpaid_arpu_change:.2f}%</div>
    </div>
    """, unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="metric-card">
        <div>Prepaid ARPU Forecast</div>
        <div class="value" style="color: {prepaid_color};">RM{prepaid_arpu_forecast_value_rounded}  {prepaid_arrow}</div>
        <div class="change" style="color: {prepaid_color};">Change: {prepaid_arpu_change:.2f}%</div>
    </div>
    """, unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="metric-card">
        <div>Blended ARPU Forecast</div>
        <div class="value" style="color: {blended_color};">RM{blended_arpu_forecast_value_rounded}  {blended_arrow}</div>
        <div class="change" style="color: {blended_color};">Change: {blended_arpu_change:.2f}%</div>
    </div>
    """, unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="metric-card">
        <div>Fibre ARPU Forecast</div>
        <div class="value" style="color: {fibre_color};">RM{fibre_arpu_forecast_value_rounded}  {fibre_arrow}</div>
        <div class="change" style="color: {fibre_color};">Change: {fibre_arpu_change:.2f}%</div>
    </div>
    """, unsafe_allow_html=True
)

# Plotly Visualizations
st.subheader('Revenue and Subscriber Trends')

fig = px.line(df, x='quarter', y=['postpaidrevenue_rmm', 'prepaidrevenue_rmm', 'wholesaleothersrevenue_rmm', 'homefibrerevenue_rmm'],
              labels={'value': 'Revenue (RM Million)', 'quarter': 'Quarter'},
              title='Revenue Trends by Segment')
st.plotly_chart(fig)

fig = px.line(df, x='quarter', y=['postpaidmobile_000', 'prepaid_000', 'fibre_000', 'totalsubscribers_000'],
              labels={'value': 'Subscribers (000)', 'quarter': 'Quarter'},
              title='Subscriber Trends by Segment')
st.plotly_chart(fig)

# Churn Prediction
st.subheader('Churn Prediction for Next Quarter')
st.write(f"Churn Prediction: {'Yes' if churn_forecast[0] == 1 else 'No'}")
st.write(f"Churn Probability: {churn_probability:.2f}")

# Adding explanatory text
st.write("""
### Interpretation of Churn Prediction
- **Churn Prediction: Yes**: This indicates that the model predicts customer churn (customers leaving) will occur in the next quarter.
- **Churn Probability: 1.00**: This means the model is 100% confident that churn will happen. While this is a strong prediction, it's essential to validate with real data and consider taking proactive measures to prevent churn.
""")

st.markdown('<h1 style="color: yellow;">Meet the BULLISH yellow man!!!</h1>', unsafe_allow_html=True)
st.video('video (2).mp4')

st.markdown('<h1 style="color: yellow;">ARPU Forecasting Methodology</h1>', unsafe_allow_html=True)

# Adding explanatory text
st.write("""

### Objective: To predict future ARPU values based on historical data.
## Technology and Methodology

### Time Series Analysis
ARPU forecasting typically uses time series analysis, which involves examining historical data points collected or recorded at specific time intervals.

### ARIMA Model
In this context, we use the ARIMA (AutoRegressive Integrated Moving Average) model. ARIMA is a popular statistical method for time series forecasting that combines:

- **Autoregression (AR)**: Uses the dependency between an observation and several lagged observations.
- **Integrated (I)**: Involves differencing the raw observations to make the time series stationary (i.e., constant mean and variance over time).
- **Moving Average (MA)**: Uses dependency between an observation and a residual error from a moving average model applied to lagged observations.

### Implementation
The ARIMA model is implemented using the `statsmodels` library in Python. The steps include:
1. Fitting the model to the historical ARPU data.
2. Generating forecasts for future periods based on the fitted model.
""")

# Display Churn Prediction Methodology
st.markdown('<h1 style="color: yellow;">Churn Prediction Methodology</h1>', unsafe_allow_html=True)

st.markdown("""


### Objective
To predict the likelihood of a customer leaving the service (churning) in the future.

### Technology and Methodology

#### Supervised Machine Learning
Churn prediction is a classification problem where the goal is to predict whether a customer will churn (leave) or not.

#### Logistic Regression
A logistic regression model is used for binary classification. It estimates the probability that a given input point belongs to a certain class (e.g., churn or no churn).

#### Features and Labels
- **Features**: Various metrics such as ARPU, revenue, subscriber counts, and net additions.
- **Label**: A binary indicator of churn, where 1 indicates the customer churned, and 0 indicates they did not.

### Implementation
Using the `sklearn` library in Python, the steps include:
1. Preparing the dataset with relevant features and labels.
2. Splitting the data into training and testing sets.
3. Training the logistic regression model on the training data.
4. Evaluating the model's accuracy on the test data.
5. Using the trained model to predict churn for future periods based on the latest data.

### Technologies Used
- **Python**: The primary programming language used for implementing the models.
- **Pandas**: For data manipulation and analysis.
- **Statsmodels**: For statistical modeling and time series analysis (ARIMA).
- **Scikit-learn (sklearn)**: For machine learning tasks, including logistic regression.
- **Streamlit**: For building the interactive web application to display the results and metrics.

### Summary
- **ARPU Forecasting**: Uses the ARIMA model from the `statsmodels` library to predict future ARPU values based on historical data. This involves time series analysis, making the data stationary, and modeling the autoregressive and moving average components.
- **Churn Prediction**: Uses logistic regression from the `sklearn` library to predict customer churn. This involves preparing features and labels, training the model on historical data, and using the model to predict future churn probabilities.
""")


# Adding a disclaimer
st.write("""
### Disclaimer
<<<<<<< HEAD
The data and analysis on this page are for demonstration purposes only prepared by [lachieng](https://lachieng.xyz). Please visit [CelcomDigi](https://celcomdigi.listedcompany.com/financials.html) for up-to-date and accurate information.
=======
The data and analysis on this page are for demonstration purposes only and prepared by lachieng. Please visit [CelcomDigi](https://celcomdigi.listedcompany.com/financials.html) for up-to-date and accurate information.
>>>>>>> 143f654ed4d6ce36c9ea8c77928c50182e9ff22d
""")
