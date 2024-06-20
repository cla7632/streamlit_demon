import streamlit as st
import pandas as pd

# URL of your background image
background_image_url = "https://images.unsplash.com/photo-1682669530322-58f634e7f5a8?q=80&w=1931&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

URL_DATA = 'https://storage.dosm.gov.my/population/population_malaysia.parquet'

df_mypop = pd.read_parquet(URL_DATA)
if 'date' in df_mypop.columns: df_mypop['date'] = pd.to_datetime(df_mypop['date'])

# Data Preparation
data = {
    "Company": ["Maxis"] * 5 + ["CelcomDigi"] * 5,
    "Quarter": ["1Q23", "2Q23", "3Q23", "4Q23", "1Q24"] * 2,
    "Postpaid Revenue (RM'm)": [864, 871, 882, 903, 911, 1283, 1267, 1257, 1268, 1249],
    "Prepaid Revenue (RM'm)": [661, 651, 652, 655, 649, 1137, 1144, 1146, 1146, 1121],
    "Home Fibre Revenue (RM'm)": [222, 229, 231, 243, 244, 40, 42, 45, 47, 46],
    "Postpaid Subscribers ('000)": [3397, 3449, 3533, 3598, 3651, 6726, 6772, 6864, 6938, 6999],
    "Prepaid Subscribers ('000)": [5686, 5684, 5682, 5875, 5771, 13459, 13592, 13614, 13483, 13322],
    "Fibre Subscribers ('000)": [688, 706, 730, 750, 765, 107, 113, 121, 131, 145],
    "Postpaid ARPU (RM)": [78.4, 77.9, 76.8, 76.5, 75.1, 69, 68, 67, 66, 64],
    "Prepaid ARPU (RM)": [38.4, 38.2, 38.1, 37.9, 37.2, 28, 28, 28, 28, 28],
    "Fibre ARPU (RM)": [108.4, 108.2, 109.5, 109.3, 110.4, 126, 127, 126, 124, 112]
}

df = pd.DataFrame(data)

# Filter the DataFrame to include only the latest quarter (1Q24)
df_latest = df[df["Quarter"] == "1Q24"]

# Apply custom CSS to style the buttons, company names, numbers, and arrows
st.markdown(
    f"""
    <style>
        .stApp {{
            background: url({background_image_url});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            height: 100vh;
            overflow: hidden;
            backdrop-filter: blur(80px); /* Blur the background image */
        }}
         .overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 1); /* Darken with 50% opacity */
        }}
        .pill-button {{
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px;
        }}

        .red-pill-button {{
            background-color: #FF5733;
        }}

        .blue-pill-button {{
            background-color: #3366FF;
        }}

        .green-text {{
            color: #28a745;
            font-size: 20px;
        }}

        .blue-text {{
            color: #007bff;
            font-size: 20px;
        }}

        .arrow-up {{
            color: #28a745;
            font-size: 20px;
        }}

        .arrow-down {{
            color: #dc3545;
            font-size: 20px;
        }}

        .arrow-right {{
            color: #f8f9fa;
            font-size: 20px;
        }}

        .percentage-up {{
            color: #28a745;
            font-size: 14px;
        }}

        .percentage-down {{
            color: #dc3545;
            font-size: 14px;
        }}

        .percentage-right {{
            color: #f8f9fa;
            font-size: 14px;
        }}

        .metric-box {{
            border: 1px solid #ccc;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            background: rgba(0, 0, 0, 1); /* Optional: Add a slight background color to the metrics */
        }}

        .metric-title {{
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .metric-result {{
            margin-top: 5px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" integrity="sha512-c0PBpyhy1v6bKzY4Q1jNL5Y5fpeZZfEUV7+O8QIZ1AelXGrMPljH2Qq/g0xUbhfSdHqk0X32cPJapAuF0Sp/1g==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    </head>
    """,
    unsafe_allow_html=True
)

# Define CSS for the icon
st.markdown(
    """
    <style>
        .icon_red {
            color: #FF5733;  /* Change this to your desired color */
            font-size: 50px; /* Adjust the font size if needed */
            margin-right: 10px; /* Optional: Adjust spacing */
        }
    </style>

    <style>
        .icon_blue {
            color: #1E90FF;  /* Change this to your desired color */
            font-size: 50px; /* Adjust the font size if needed */
            margin-right: 10px; /* Optional: Adjust spacing */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Content within the overlay
st.markdown('<div class="overlay"></div>', unsafe_allow_html=True)

# Streamlit Application
st.title("CelcomDigi's Journey (Quarter [1Q24]): Red Pill or Blue Pill?")

# Helper function to get arrow and percentage based on comparison
def get_arrow(current, previous):
    if current > previous:
        percentage = ((current - previous) / previous) * 100
        return f'<i class="fas fa-arrow-up arrow-up"></i> <span class="percentage-up">{percentage:.2f}%</span>'
    elif current < previous:
        percentage = ((previous - current) / previous) * 100
        return f'<i class="fas fa-arrow-down arrow-down"></i> <span class="percentage-down">{percentage:.2f}%</span>'
    else:
        return '<i class="fas fa-arrow-right arrow-right"></i> <span class="percentage-right">0.00%</span>'

# Function to display results
def show_results(company_name, superior_data, previous_data, competitor_data, competitor_previous_data, company_color, competitor_color):
    st.markdown(f"### {company_name} Comparison")
    if not superior_data.empty:
        st.write(f"{company_name} outperforms its competitor in the following areas:")
        columns = st.columns(2)
        for i, (metric, value) in enumerate(superior_data.items()):
            company_arrow = get_arrow(value, previous_data[metric])
            competitor_arrow = get_arrow(competitor_data[metric], competitor_previous_data[metric])
            competitor_name = "Maxis" if company_name == "CelcomDigi" else "CelcomDigi"
            with columns[i % 2]:
                st.write(f"<div class='metric-box'><div class='metric-title'>{metric}</div><div class='metric-result'><span class='{company_color}'>{company_name}: {value} {company_arrow}</span><br/><span class='{competitor_color}'>{competitor_name}: {competitor_data[metric]} {competitor_arrow}</span></div></div>", unsafe_allow_html=True)
    else:
        st.write(f"No metrics where {company_name} outperforms its competitor.")

# Initialize session state for the pill choice if it doesn't exist
if "pill" not in st.session_state:
    st.session_state.pill = None

# Buttons for choosing Red or Blue Pill
col1, col2 = st.columns(2)

with col1:
    if st.button("Red Pill", key="red_pill", help="Red Pill"):
        st.markdown('<i class="fas fa-pills icon_red"></i>', unsafe_allow_html=True)
        st.session_state.pill = 'red'

with col2:
    if st.button("Blue Pill", key="blue_pill", help="Blue Pill"):
        st.markdown('<i class="fas fa-pills icon_blue"></i>', unsafe_allow_html=True)
        st.session_state.pill = 'blue'

# Determine which pill was chosen and display corresponding data
if st.session_state.pill == 'red':
    # Filter Maxis' data for the latest quarter
    maxis_data = df_latest[df_latest["Company"] == "Maxis"].select_dtypes(include="number").mean()
    celcom_data = df_latest[df_latest["Company"] == "CelcomDigi"].select_dtypes(include="number").mean()
    maxis_superior = maxis_data[maxis_data > celcom_data].dropna()
    
    # Get data for the previous quarter (4Q23)
    df_previous = df[df["Quarter"] == "4Q23"]
    maxis_previous = df_previous[df_previous["Company"] == "Maxis"].select_dtypes(include="number").mean()
    celcom_previous = df_previous[df_previous["Company"] == "CelcomDigi"].select_dtypes(include="number").mean()

    show_results("Maxis", maxis_superior, maxis_previous, celcom_data, celcom_previous, "green-text", "blue-text")

elif st.session_state.pill == 'blue':
    # Filter CelcomDigi's data for the latest quarter
    maxis_data = df_latest[df_latest["Company"] == "Maxis"].select_dtypes(include="number").mean()
    celcom_data = df_latest[df_latest["Company"] == "CelcomDigi"].select_dtypes(include="number").mean()
    celcom_superior = celcom_data[celcom_data > maxis_data].dropna()
    
    # Get data for the previous quarter (4Q23)
    df_previous = df[df["Quarter"] == "4Q23"]
    maxis_previous = df_previous[df_previous["Company"] == "Maxis"].select_dtypes(include="number").mean()
    celcom_previous = df_previous[df_previous["Company"] == "CelcomDigi"].select_dtypes(include="number").mean()

    show_results("CelcomDigi", celcom_superior, celcom_previous, maxis_data, maxis_previous, "blue-text", "green-text")

# Display the DataFrame as a table
st.markdown("### Latest Quarter Data (1Q24)")
st.dataframe(df)

st.dataframe(df_mypop)
