import pandas as pd
import streamlit as st
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

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
latest_postpaid_arpu = float(df['postpaidmobilearpu_rm'].iloc[-1])
latest_prepaid_arpu = float(df['prepaidarpu_rm'].iloc[-1])
latest_blended_arpu = float(df['blendedmobilearpu_rm'].iloc[-1])
latest_fibre_arpu = float(df['fibrearpu_rm'].iloc[-1])

postpaid_arpu_change = float(((postpaid_arpu_forecast.iloc[0] - latest_postpaid_arpu) / latest_postpaid_arpu) * 100)
prepaid_arpu_change = float(((prepaid_arpu_forecast.iloc[0] - latest_prepaid_arpu) / latest_prepaid_arpu) * 100)
blended_arpu_change = float(((blended_arpu_forecast.iloc[0] - latest_blended_arpu) / latest_blended_arpu) * 100)
fibre_arpu_change = float(((fibre_arpu_forecast.iloc[0] - latest_fibre_arpu) / latest_fibre_arpu) * 100)

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

st.subheader('Latest Quarter(Q42023) Information')

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
    arrow = "↑" if change > 0 else "↓" if change < 0 else ""
    change_text = f" ({change:.2f}%) {arrow}"
    return change_text

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Revenue Metrics (Left Column)
with col1:
    postpaid_revenue_color = get_color(latest_quarter['postpaidrevenue_rmm'], previous_quarter['postpaidrevenue_rmm'])
    prepaid_revenue_color = get_color(latest_quarter['prepaidrevenue_rmm'], previous_quarter['prepaidrevenue_rmm'])
    postpaid_revenue_change = get_change_and_arrow(latest_quarter['postpaidrevenue_rmm'], previous_quarter['postpaidrevenue_rmm'])
    prepaid_revenue_change = get_change_and_arrow(latest_quarter['prepaidrevenue_rmm'], previous_quarter['prepaidrevenue_rmm'])
    st.markdown(
        f"""
        <div class="metric-card">
            <div>Postpaid Revenue (RM'm): <span class="value" style="color:{postpaid_revenue_color}">{latest_quarter['postpaidrevenue_rmm']:,}{postpaid_revenue_change}</span></div>
            <div>Prepaid Revenue (RM'm): <span class="value" style="color:{prepaid_revenue_color}">{latest_quarter['prepaidrevenue_rmm']:,}{prepaid_revenue_change}</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Subscribers Metrics (Right Column)
with col2:
    postpaid_subscribers_color = get_color(latest_quarter['postpaidmobile_000'], previous_quarter['postpaidmobile_000'])
    prepaid_subscribers_color = get_color(latest_quarter['prepaid_000'], previous_quarter['prepaid_000'])
    fibre_subscribers_color = get_color(latest_quarter['fibre_000'], previous_quarter['fibre_000'])
    total_subscribers_color = get_color(latest_quarter['totalsubscribers_000'], previous_quarter['totalsubscribers_000'])
    postpaid_subscribers_change = get_change_and_arrow(latest_quarter['postpaidmobile_000'], previous_quarter['postpaidmobile_000'])
    prepaid_subscribers_change = get_change_and_arrow(latest_quarter['prepaid_000'], previous_quarter['prepaid_000'])
    fibre_subscribers_change = get_change_and_arrow(latest_quarter['fibre_000'], previous_quarter['fibre_000'])
    total_subscribers_change = get_change_and_arrow(latest_quarter['totalsubscribers_000'], previous_quarter['totalsubscribers_000'])
    st.markdown(
        f"""
        <div class="metric-card">
            <div>Total Postpaid Subscribers ('000): <span class="value" style="color:{postpaid_subscribers_color}">{latest_quarter['postpaidmobile_000']:,}{postpaid_subscribers_change}</span></div>
            <div>Total Prepaid Subscribers ('000): <span class="value" style="color:{prepaid_subscribers_color}">{latest_quarter['prepaid_000']:,}{prepaid_subscribers_change}</span></div>
            <div>Total Fibre Subscribers ('000): <span class="value" style="color:{fibre_subscribers_color}">{latest_quarter['fibre_000']:,}{fibre_subscribers_change}</span></div>
            <div>Total Subscribers ('000): <span class="value" style="color:{total_subscribers_color}">{latest_quarter['totalsubscribers_000']:,}{total_subscribers_change}</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Net Add Metrics (Left Column)
with col3:
    netadd_postpaid_color = get_color(latest_quarter['netaddpostpaid_000'], previous_quarter['netaddpostpaid_000'])
    netadd_prepaid_color = get_color(latest_quarter['netaddprepaid_000'], previous_quarter['netaddprepaid_000'])
    netadd_fibre_color = get_color(latest_quarter['netaddfibre_000'], previous_quarter['netaddfibre_000'])
    netadd_postpaid_change = get_change_and_arrow(latest_quarter['netaddpostpaid_000'], previous_quarter['netaddpostpaid_000'])
    netadd_prepaid_change = get_change_and_arrow(latest_quarter['netaddprepaid_000'], previous_quarter['netaddprepaid_000'])
    netadd_fibre_change = get_change_and_arrow(latest_quarter['netaddfibre_000'], previous_quarter['netaddfibre_000'])
    st.markdown(
        f"""
        <div class="metric-card">
            <div>Net Add Postpaid ('000): <span class="value" style="color:{netadd_postpaid_color}">{latest_quarter['netaddpostpaid_000']:,}{netadd_postpaid_change}</span></div>
            <div>Net Add Prepaid ('000): <span class="value" style="color:{netadd_prepaid_color}">{latest_quarter['netaddprepaid_000']:,}{netadd_prepaid_change}</span></div>
            <div>Net Add Fibre ('000): <span class="value" style="color:{netadd_fibre_color}">{latest_quarter['netaddfibre_000']:,}{netadd_fibre_change}</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ARPU Metrics (Right Column)
with col4:
    postpaid_arpu_color = get_color(latest_quarter['postpaidmobilearpu_rm'], previous_quarter['postpaidmobilearpu_rm'])
    prepaid_arpu_color = get_color(latest_quarter['prepaidarpu_rm'], previous_quarter['prepaidarpu_rm'])
    blended_arpu_color = get_color(latest_quarter['blendedmobilearpu_rm'], previous_quarter['blendedmobilearpu_rm'])
    fibre_arpu_color = get_color(latest_quarter['fibrearpu_rm'], previous_quarter['fibrearpu_rm'])
    postpaid_arpu_change = get_change_and_arrow(latest_quarter['postpaidmobilearpu_rm'], previous_quarter['postpaidmobilearpu_rm'])
    prepaid_arpu_change = get_change_and_arrow(latest_quarter['prepaidarpu_rm'], previous_quarter['prepaidarpu_rm'])
    blended_arpu_change = get_change_and_arrow(latest_quarter['blendedmobilearpu_rm'], previous_quarter['blendedmobilearpu_rm'])
    fibre_arpu_change = get_change_and_arrow(latest_quarter['fibrearpu_rm'], previous_quarter['fibrearpu_rm'])
    st.markdown(
        f"""
        <div class="metric-card">
            <div>Postpaid Mobile ARPU (RM): <span class="value" style="color:{postpaid_arpu_color}">{latest_quarter['postpaidmobilearpu_rm']:,}{postpaid_arpu_change}</span></div>
            <div>Prepaid ARPU (RM): <span class="value" style="color:{prepaid_arpu_color}">{latest_quarter['prepaidarpu_rm']:,}{prepaid_arpu_change}</span></div>
            <div>Blended Mobile ARPU (RM): <span class="value" style="color:{blended_arpu_color}">{latest_quarter['blendedmobilearpu_rm']:,}{blended_arpu_change}</span></div>
            <div>Fibre ARPU (RM): <span class="value" style="color:{fibre_arpu_color}">{latest_quarter['fibrearpu_rm']:,}{fibre_arpu_change}</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Plotly Interactive Charts
st.subheader('Revenue Over Time')
fig = px.line(df, x='quarter', y=["postpaidrevenue_rmm", "prepaidrevenue_rmm", "wholesaleothersrevenue_rmm", "homefibrerevenue_rmm"])
st.plotly_chart(fig)

st.subheader('ARPU Over Time')
fig_arpu = px.line(df, x='quarter', y=["postpaidmobilearpu_rm", "prepaidarpu_rm", "blendedmobilearpu_rm", "fibrearpu_rm", "blendedarpu_rm"])
st.plotly_chart(fig_arpu)

# Display DataFrame
st.dataframe(df)

# Adding the forecasts to the Streamlit dashboard
st.subheader('ARPU Forecast for Next Quarter')

# Check the type of postpaid_arpu_forecast.iloc[0]
print(type(postpaid_arpu_forecast.iloc[0]))
print(type(postpaid_arpu_change))

# Remove percentage and arrow symbols from postpaid_arpu_change
postpaid_arpu_change = postpaid_arpu_change.replace('(', '').replace(')', '').replace('↓', '')


# Convert them to float if they are not already numerical
postpaid_arpu_forecast_value = float(postpaid_arpu_forecast.iloc[0])
postpaid_arpu_change_value = float(postpaid_arpu_change)

# Display the values with formatting
st.write(f"Postpaid ARPU: {postpaid_arpu_forecast_value:.2f} RM ({postpaid_arpu_change_value:.2f}% change)")

st.write(f"Postpaid ARPU: {postpaid_arpu_forecast.iloc[0]:.2f} RM ({postpaid_arpu_change:.2f}% change)")
st.write(f"Prepaid ARPU: {prepaid_arpu_forecast.iloc[0]:.2f} RM ({prepaid_arpu_change:.2f}% change)")
st.write(f"Blended ARPU: {blended_arpu_forecast.iloc[0]:.2f} RM ({blended_arpu_change:.2f}% change)")
st.write(f"Fibre ARPU: {fibre_arpu_forecast.iloc[0]:.2f} RM ({fibre_arpu_change:.2f}% change)")

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

# Adding a disclaimer
st.write("""
### Disclaimer
The data and analysis on this page are for demonstration purposes only. Please visit [CelcomDigi](https://celcomdigi.listedcompany.com/financials.html) for up-to-date and accurate information.
""")
