import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Sample data (replace with actual data)
data = {
    "Company": ["Maxis"] * 9 + ["CelcomDigi"] * 9,
    "Quarter": ["1Q22", "2Q22", "3Q22", "4Q22", "1Q23", "2Q23", "3Q23", "4Q23", "1Q24"] * 2,
    "Postpaid Revenue (RM'm)": [785, 810, 830, 850, 864, 871, 882, 903, 911, 1270, 1272, 1287, 1290, 1283, 1267, 1257, 1268, 1249],
    "Prepaid Revenue (RM'm)": [657, 679, 676, 681, 661, 651, 652, 655, 649, 1118, 1145, 1125, 1148, 1137, 1144, 1146, 1146, 1121],
    "Home Fibre Revenue (RM'm)": [200, 209, 215, 219, 222, 229, 231, 243, 244, 31, 33, 35, 37, 40, 42, 45, 47, 46]
}

df = pd.DataFrame(data)

# Streamlit App
st.title("Revenue Comparison: Maxis vs. CelcomDigi")

# Sidebar for user inputs
st.sidebar.header("Filter")
selected_metric = st.sidebar.selectbox("Select Metric", ["Postpaid Revenue (RM'm)", "Prepaid Revenue (RM'm)", "Home Fibre Revenue (RM'm)"])

# Plotting the selected metric for both companies
fig = px.line(df, x="Quarter", y=selected_metric, color="Company", title=f"Revenue Comparison: Maxis vs. CelcomDigi - {selected_metric}")
st.plotly_chart(fig)

# Prepare data for machine learning
X = pd.get_dummies(df[['Quarter', 'Company']], drop_first=True)
y = df[selected_metric]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions for the next quarter (1Q24)
next_quarter = pd.DataFrame({'Quarter': ['1Q24', '1Q24'], 'Company': ['Maxis', 'CelcomDigi']})
X_next = pd.get_dummies(next_quarter, drop_first=True)

# Ensure X_next has the same columns as X_train for prediction
X_next = X_next.reindex(columns=X_train.columns, fill_value=0)

# Predict for next quarter
predicted_revenue = model.predict(X_next)

# Display model predictions and actual values for 1Q24
actual_1Q24_revenue_maxis = df[(df["Company"] == "Maxis") & (df["Quarter"] == "1Q24")][selected_metric].values[0]
actual_1Q24_revenue_celcomdigi = df[(df["Company"] == "CelcomDigi") & (df["Quarter"] == "1Q24")][selected_metric].values[0]

st.subheader("Model Predictions and Actual Values")
predicted_df = pd.DataFrame({
    'Company': ['Maxis', 'CelcomDigi'],
    'Quarter': ['1Q24', '1Q24'],
    'Predicted Revenue': predicted_revenue,
    'Actual Revenue': [actual_1Q24_revenue_maxis, actual_1Q24_revenue_celcomdigi]
})
st.dataframe(predicted_df)

# Model Interpretation
st.subheader("Model Interpretation")
st.write(f"The model used is Linear Regression, which predicts {selected_metric} based on historical quarters and company data.")
st.write("The model suggests that future revenue trends can be projected based on historical data patterns.")

# Additional Insights
st.subheader("Key Insights")
max_revenue_maxis = df[df["Company"] == "Maxis"][selected_metric].max()
max_quarter_maxis = df[(df["Company"] == "Maxis") & (df[selected_metric] == max_revenue_maxis)]["Quarter"].values[0]
st.write(f"The highest {selected_metric} recorded by Maxis was {max_revenue_maxis} RM'm in {max_quarter_maxis}.")

max_revenue_celcomdigi = df[df["Company"] == "CelcomDigi"][selected_metric].max()
max_quarter_celcomdigi = df[(df["Company"] == "CelcomDigi") & (df[selected_metric] == max_revenue_celcomdigi)]["Quarter"].values[0]
st.write(f"The highest {selected_metric} recorded by CelcomDigi was {max_revenue_celcomdigi} RM'm in {max_quarter_celcomdigi}.")

# Display raw data
st.subheader("Raw Data")
st.dataframe(df)


# Link to the other app
st.markdown("My 1st interview demo, visit [here](https://lamachinelearningdemo.streamlit.app/)")
st.markdown("For a detailed comparison, visit [here](https://cmdcomparison.streamlit.app/)")
