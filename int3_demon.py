import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from st_aggrid import AgGrid, GridOptionsBuilder

# Sample data (replace with actual data)
data = {
    "Company": ["Maxis"] * 9 + ["CelcomDigi"] * 9,
    "Quarter": ["1Q22", "2Q22", "3Q22", "4Q22", "1Q23", "2Q23", "3Q23", "4Q23", "1Q24"] * 2,
    "Postpaid Revenue (RM'm)": [785, 810, 830, 850, 864, 871, 882, 903, 911, 1270, 1272, 1287, 1290, 1283, 1267, 1257, 1268, 1249],
    "Prepaid Revenue (RM'm)": [657, 679, 676, 681, 661, 651, 652, 655, 649, 1118, 1145, 1125, 1148, 1137, 1144, 1146, 1146, 1121],
    "Home Fibre Revenue (RM'm)": [200, 209, 215, 219, 222, 229, 231, 243, 244, 31, 33, 35, 37, 40, 42, 45, 47, 46]
}

df = pd.DataFrame(data)

# Adding a disclaimer
st.write("""
### Disclaimer

The data and analysis on this page are for demonstration purposes only and prepared by [lachieng](https://lachieng.xyz). Please visit [CelcomDigi](https://celcomdigi.listedcompany.com/financials.html), and [Maxis](https://maxis.listedcompany.com/financials.html) for up-to-date and accurate information.

""")

# Streamlit App
st.title("Revenue Comparison: Maxis vs. CelcomDigi")

# Sidebar for user inputs
st.sidebar.header("Filter")
selected_metric = st.sidebar.selectbox("Select Metric", ["Postpaid Revenue (RM'm)", "Prepaid Revenue (RM'm)", "Home Fibre Revenue (RM'm)"])

# Prepare data for machine learning
X = pd.get_dummies(df[['Quarter', 'Company']], drop_first=True)
y = df[selected_metric]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Make predictions for the next quarter (1Q24)
next_quarter = pd.DataFrame({'Quarter': ['1Q24', '1Q24'], 'Company': ['Maxis', 'CelcomDigi']})
X_next = pd.get_dummies(next_quarter, drop_first=True)

# Ensure X_next has the same columns as X_train for prediction
X_next = X_next.reindex(columns=X_train.columns, fill_value=0)

# Predict for next quarter
predicted_revenue = model.predict(X_next)

# Extend the DataFrame with predicted values
predicted_df = pd.DataFrame({
    'Company': ['Maxis', 'CelcomDigi'],
    'Quarter': ['1Q24', '1Q24'],
    selected_metric: predicted_revenue
})

# Combine actual and predicted data for plotting
combined_df = pd.concat([df, predicted_df], ignore_index=True)

# Plotting the selected metric for both companies
fig = go.Figure()

for company in combined_df['Company'].unique():
    company_data = combined_df[combined_df['Company'] == company]
    
    color = 'green' if company == 'Maxis' else 'blue'
    
    fig.add_trace(go.Scatter(
        x=company_data['Quarter'],
        y=company_data[selected_metric],
        mode='lines+markers',
        name=f'{company}',
        line=dict(color=color)
    ))
    
    # Add dotted line for predicted values in orange
    if company == 'Maxis':
        fig.add_trace(go.Scatter(
            x=['4Q23', '1Q24'],
            y=[company_data[company_data['Quarter'] == '4Q23'][selected_metric].values[0], predicted_revenue[0]],
            mode='lines+markers',
            name=f'{company} Predicted',
            line=dict(color='orange', dash='dot')
        ))
    elif company == 'CelcomDigi':
        fig.add_trace(go.Scatter(
            x=['4Q23', '1Q24'],
            y=[company_data[company_data['Quarter'] == '4Q23'][selected_metric].values[0], predicted_revenue[1]],
            mode='lines+markers',
            name=f'{company} Predicted',
            line=dict(color='yellow', dash='dot')
        ))

fig.update_layout(title=f"Revenue Comparison: Maxis vs. CelcomDigi - {selected_metric}",
                xaxis_title='Quarter',
                yaxis_title=selected_metric)

st.plotly_chart(fig)

# Display model predictions and actual values for 1Q24
actual_1Q24_revenue_maxis = df[(df["Company"] == "Maxis") & (df["Quarter"] == "1Q24")][selected_metric].values[0]
actual_1Q24_revenue_celcomdigi = df[(df["Company"] == "CelcomDigi") & (df["Quarter"] == "1Q24")][selected_metric].values[0]

st.subheader("Model Predictions and Actual Values")
predicted_comparison_df = pd.DataFrame({
    'Company': ['Maxis', 'CelcomDigi'],
    'Quarter': ['1Q24', '1Q24'],
    'Predicted Revenue': predicted_revenue,
    'Actual Revenue': [actual_1Q24_revenue_maxis, actual_1Q24_revenue_celcomdigi]
})
st.dataframe(predicted_comparison_df)

# Model Interpretation
st.subheader("Model Interpretation")
st.write(f"The model used is Linear Regression, which predicts {selected_metric} based on historical quarters data.")
st.write(f"The Mean Absolute Error (MAE) of the model on the test set is: {mae:.2f} RM'm.")
st.write("The model suggests that future revenue trends can be projected based on historical data patterns.")

# Additional Insights
st.subheader("Key Insights")
max_revenue_maxis = df[df["Company"] == "Maxis"][selected_metric].max()
max_quarter_maxis = df[(df["Company"] == "Maxis") & (df[selected_metric] == max_revenue_maxis)]["Quarter"].values[0]
st.write(f"The highest {selected_metric} recorded by Maxis was {max_revenue_maxis} RM'm in {max_quarter_maxis}.")

max_revenue_celcomdigi = df[df["Company"] == "CelcomDigi"][selected_metric].max()
max_quarter_celcomdigi = df[(df["Company"] == "CelcomDigi") & (df[selected_metric] == max_revenue_celcomdigi)]["Quarter"].values[0]
st.write(f"The highest {selected_metric} recorded by CelcomDigi was {max_revenue_celcomdigi} RM'm in {max_quarter_celcomdigi}.")

# Display raw data with AG Grid
st.subheader("Raw Data with AG Grid")
gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_pagination(paginationAutoPageSize=True)
gridOptions = gb.build()
AgGrid(df, gridOptions=gridOptions)


st.markdown("My 1st interview [demo](https://lamachinelearningdemo.streamlit.app/). For more, visit [here](https://cmdcomparison.streamlit.app/)")

