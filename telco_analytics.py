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

# https://storage.dosm.gov.my/gdp/gdp_2024-q1.pdf, page 20
new_gdp_data = {
    "quarter": ["1Q2022", "2Q2022", "3Q2022", "4Q2022", "1Q2023", "2Q2023", "3Q2023", "4Q2023","1Q2024"],
    "Malaysia_GDP(%)": [5.1, 8.9, 14.4, 7.4, 5.5, 2.8, 3.1, 2.9,4.2]
}

# CPI data from https://storage.dosm.gov.my/cpi/cpi_2024-04_main.xlsx
cpi_data = {
    "month": [
        "Jan2023", "Feb2023", "Mar2023", "Apr2023", "May2023", "Jun2023", 
        "Jul2023", "Aug2023", "Sept2023", "Oct2023", "Nov2023", "Dec2023", 
        "Jan2024", "Feb2024", "Mae2024", "Apr2024"
    ],
    "CPI(RM)": [
        130.7, 131.2, 131.2, 131.3, 131.5, 131.7, 131.8, 132.0, 132.0, 132.1, 
        132.2, 132.4, 132.7, 133.4, 133.4, 133.6
    ],
    "Information&Communication_CPI(RM)": [
        96.0, 96.0, 96.0, 96.0, 93.7, 93.7, 93.6, 93.6, 93.7, 93.6, 
        93.7, 93.6, 93.6, 93.6, 93.6, 93.5
    ]
}

# based on BNM official "https://www.bnm.gov.my/monetary-stability/opr-decisions"
opr_data = {
    "date": [
        "20 Jan 2022", "03 Mar 2022", "11 May 2022", "06 Jul 2022", 
        "08 Sep 2022", "03 Nov 2022", "19 Jan 2023", "09 Mar 2023", 
        "03 May 2023", "06 Jul 2023", "07 Sep 2023", "02 Nov 2023", 
        "24 Jan 2024", "07 Mar 2024", "09 May 2024"
    ],
    "change_in_opr(%)": [
        0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0
    ],
    "new_opr_level(%)": [
        1.75, 1.75, 2, 2.25, 2.5, 2.75, 2.75, 2.75, 3, 3, 3, 3, 3, 3, 3
    ]
}


df_gdp = pd.DataFrame(new_gdp_data)

df = pd.DataFrame(data)

df_cpi = pd.DataFrame(cpi_data)

df_opr = pd.DataFrame(opr_data)


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

# Adding a disclaimer
st.write("""
### Disclaimer

The data and analysis on this page are for demonstration purposes only and prepared by [lachieng](https://lachieng.xyz). Please visit [CelcomDigi](https://celcomdigi.listedcompany.com/financials.html), [DOSM](https://www.dosm.gov.my/), and [BNM](https://www.bnm.gov.my/monetary-stability/opr-decisions) for up-to-date and accurate information.

""")


# Streamlit Dashboard
st.markdown('<h1 style="color: yellow;">Telco ARPU Analytics & Prediction</h1>', unsafe_allow_html=True)

# Metric Cards for Latest Quarter Information
latest_quarter = df.iloc[-1]
previous_quarter = df.iloc[-2]

# Find the latest quarter from the DataFrame
latest_Q = df['quarter'].iloc[-1]

st.subheader(f"Latest Quarter {latest_Q} Information")


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

st.subheader("Revenue Metrics")
display_metric_cards(revenue_metrics)

st.subheader("ARPU Metrics")
display_metric_cards(arpu_metrics)

st.subheader("Subscriber Metrics")
display_metric_cards(subscriber_metrics)

st.subheader("Net Adds Metrics")
display_metric_cards(net_adds_metrics)

# ARPU Forecast
st.markdown('<h3 style="display: inline-block; border-radius: 5px; background-color: gray; color: white;">ARPU Forecast for Next Quarter</h3>', unsafe_allow_html=True)


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



# Churn Prediction

# st.markdown('<h3 style="display: inline-block; border-radius: 5px; background-color: gray; color: white;">Churn Prediction for Next Quarter</h3>', unsafe_allow_html=True)


# st.write(f"Churn Prediction: {'Yes' if churn_forecast[0] == 1 else 'No'}")
# st.write(f"Churn Probability: {churn_probability:.2f}")

# Adding explanatory text
# st.write("""
# ### Interpretation of Churn Prediction
# - **Churn Prediction: Yes**: This indicates that the model predicts customer churn (customers leaving) will occur in the next quarter.
# - **Churn Probability: 1.00**: This means the model is 100% confident that churn will happen. While this is a strong prediction, it's essential to validate with real data and consider taking proactive measures to prevent churn.
# """)

# Display the DataFrame itself for reference
st.subheader('DataSet')
st.write(df)
st.write("""

Source: CelcomDigi Financial reports
""")
# Plotly Visualizations
st.subheader('ARPU, Revenue and Subscriber Trends')

# ARPU Trends


fig_line_ARPU = px.line(df, x='quarter', y=['postpaidmobilearpu_rm', 'prepaidarpu_rm', 'blendedmobilearpu_rm', 'fibrearpu_rm','blendedarpu_rm'],
                                     labels={'value': 'ARPU (RM)', 'quarter': 'Quarter'},
                                     title='ARPU Trends by Segment (Line Plot)',
                                     template="plotly_dark")
st.plotly_chart(fig_line_ARPU)

# Revenue Trends


fig_bar_revenue = px.bar(df, x='quarter', y=['postpaidrevenue_rmm', 'prepaidrevenue_rmm', 'wholesaleothersrevenue_rmm', 'homefibrerevenue_rmm'],
                         labels={'value': 'Revenue (RM Million)', 'quarter': 'Quarter'},
                         title='Revenue Trends by Segment (Bar Chart)',
                         template="plotly_dark",
                         barmode='group')
st.plotly_chart(fig_bar_revenue)



# Subscriber Trends


fig_scatter_subscribers = px.scatter(df, x='quarter', y=['postpaidmobile_000', 'prepaid_000', 'fibre_000', 'totalsubscribers_000'],
                                     labels={'value': 'Subscribers (000)', 'quarter': 'Quarter'},
                                     title='Subscriber Trends by Segment (Scatter Plot)',
                                     template="plotly_dark")
st.plotly_chart(fig_scatter_subscribers)


st.subheader('External trends like GDP, CPI, and OPR')
# GDP Trends


fig_line_GDP = px.line(df_gdp, x='quarter', y=['Malaysia_GDP(%)'],
                                     labels={'value': 'Percentage', 'quarter': 'Quarter'},
                                     title='GDP growth (Line Plot)',
                                     template="plotly_dark")
st.plotly_chart(fig_line_GDP)
st.write("""

Source: Department of Statistics Malaysia
""")
# CPI Trends

fig_scatter_CPI = px.scatter(df_cpi, x='month', y=['CPI(RM)','Information&Communication_CPI(RM)'],
                                     labels={'value': 'RM', 'month': 'Monthly'},
                                     title='CPI (Scatter Plot)',
                                     template="plotly_dark")
st.plotly_chart(fig_scatter_CPI)
st.write("""

Source: Department of Statistics Malaysia
""")

# OPR Trends


fig_line_OPR = px.line(df_opr, x='date', y=['change_in_opr(%)','new_opr_level(%)'],
                                     labels={'value': 'Percentage', 'date': 'Date'},
                                     title='OPR (Line Plot)',
                                     template="plotly_dark")
st.plotly_chart(fig_line_OPR)

st.write("""

Source: Bank Negara Malaysia
""")
# Display the DataFrame itself for reference
# st.subheader('DataSet')
# st.write(df)

# st.subheader('DataSet(GDP)')
# st.write(df_gdp)

# st.subheader('DataSet(CPI)')
# st.write(df_cpi)

# st.subheader('DataSet(OPR)')
# st.write(df_opr)





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
# st.markdown('<h1 style="color: yellow;">Churn Prediction Methodology</h1>', unsafe_allow_html=True)

st.markdown("""


### Technologies Used
- **Python**: The primary programming language used for implementing the models.
- **Pandas**: For data manipulation and analysis.
- **Statsmodels**: For statistical modeling and time series analysis (ARIMA).
- **Streamlit**: For building the interactive web application to display the results and metrics.

### Summary
- **ARPU Forecasting**: Uses the ARIMA model from the `statsmodels` library to predict future ARPU values based on historical data. This involves time series analysis, making the data stationary, and modeling the autoregressive and moving average components.
""")


