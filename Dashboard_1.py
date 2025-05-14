import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Page config
st.set_page_config(page_title="Sales Trends Dashboard", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    h1, h2, h3 { color: #1f77b4; }
    .stButton>button { background-color: #4CAF50; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("🛍️ AI Sales Trends Intelligence Dashboard")

# Load model and data
model = joblib.load("Sales_Trends_prediction_model_rf.pkl")
df = pd.read_csv("Multiclass Clothing Sales Dataset.csv")

# Encoding mappings
payment_method_map = {"Card": 0, "UPI": 1, "Cash": 2}
brand_map = {'Forever 21':0,'Ralph Lauren':1,'Nike':2,'Zara':3,'Wrangler':4,'Balenciaga':5,
                                       'Uniqlo':6,'Reebok':7,'Gucci':8,'Louis Vuitton':9,'Tommy Hilfiger':10,'Under Armour':11,
                                       'Calvin Klein':12,'Puma':13,'Diesel':14,'Versace':15,'Levi’s':16,'Adidas':17,'H&M':18,'GAP':19}
season_map = {"Summer": 0, "Winter": 1, "All season": 2}

# ----------------------- Prediction Section -----------------------
st.header("🔮 Predict Selling Price")
with st.expander("Enter product and customer details"):

    col1, col2, col3 = st.columns(3)

    with col1:
        profit_margin = st.number_input("Profit Margin", value=0.0)
        cost_price = st.number_input("Cost Price", value=0.0)
        purchase_frequency = st.number_input("Purchase Frequency", value=0.0)
        store_rating = st.number_input("Store Rating", value=0.0)
        price_elasticity = st.number_input("Price Elasticity", value=0.0)

    with col2:
        demand_index = st.number_input("Demand Index", value=0.0)
        customer_age = st.number_input("Customer Age", value=25)
        total_sales = st.number_input("Total Sales", value=0.0)
        return_rate = st.number_input("Return Rate", value=0.0)
        discount_percentage = st.number_input("Discount Percentage", value=0.0)

    with col3:
        stock_availability = st.number_input("Stock Availability", value=1)
        payment_method_str = st.selectbox("Payment Method", list(payment_method_map.keys()))
        brand_str = st.selectbox("Brand", list(brand_map.keys()))
        season_str = st.selectbox("Season", list(season_map.keys()))
        quantity_sold = st.number_input("Quantity Sold", value=1)

        # Encode strings
        payment_method = payment_method_map[payment_method_str]
        brand = brand_map[brand_str]
        season = season_map[season_str]

    features = np.array([[profit_margin, cost_price, purchase_frequency, store_rating,
                          price_elasticity, demand_index, customer_age, total_sales,
                          return_rate, discount_percentage, stock_availability,
                          payment_method, brand, season, quantity_sold]])

    if st.button("💡 Predict optimal Price"):
        prediction = model.predict(features)
        st.success(f"🎯 Predicted Optimal Price: ₹{prediction[0]:,.2f}")

# ----------------------- Sidebar Filters -----------------------
st.sidebar.header("🔍 Filter Dataset to Visualize Trends")
brands = st.sidebar.multiselect("Select Brands", options=df["Brand"].unique(), default=df["Brand"].unique())
seasons = st.sidebar.multiselect("Select Seasons", options=df["Season"].unique(), default=df["Season"].unique())
payments = st.sidebar.multiselect("Select Payment Methods", options=df["Payment_Method"].unique(), default=df["Payment_Method"].unique())

age_range = st.sidebar.slider("Customer Age", int(df["Customer_Age"].min()), int(df["Customer_Age"].max()), (20, 60))
quantity_range = st.sidebar.slider("Quantity Sold", int(df["Quantity_Sold"].min()), int(df["Quantity_Sold"].max()), (1, 100))

# Apply filters
filtered_df = df[
    (df["Brand"].isin(brands)) &
    (df["Season"].isin(seasons)) &
    (df["Payment_Method"].isin(payments)) &
    (df["Customer_Age"].between(*age_range)) &
    (df["Quantity_Sold"].between(*quantity_range))
]

# ----------------------- Visualization -----------------------
st.markdown("---")
st.header("📈 Real-Time Sales Data Trends")

if filtered_df.empty:
    st.warning("⚠️ No data available for selected filters.")
else:
    # Top metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("📦 Avg Selling Price", f"₹{filtered_df['Selling_Price'].mean():,.2f}")
    col2.metric("🔁 Avg Return Rate", f"{filtered_df['Return_Rate'].mean():.2f}")
    col3.metric("📊 Total Sales", f"{filtered_df['Total_Sales'].sum():,.0f}")

    # Distribution charts
    st.subheader("🧮 Metric Distributions")
    dist1, dist2, dist3 = st.columns(3)

    with dist1:
        fig1, ax1 = plt.subplots()
        sns.histplot(filtered_df["Selling_Price"], kde=True, color="#1f77b4", ax=ax1)
        ax1.set_title("Selling Price")
        st.pyplot(fig1)

    with dist2:
        fig2, ax2 = plt.subplots()
        sns.histplot(filtered_df["Discount_Percentage"], kde=True, color="#ff7f0e", ax=ax2)
        ax2.set_title("Discount Percentage")
        st.pyplot(fig2)

    with dist3:
        fig3, ax3 = plt.subplots()
        sns.histplot(filtered_df["Quantity_Sold"], kde=True, color="#2ca02c", ax=ax3)
        ax3.set_title("Quantity Sold")
        st.pyplot(fig3)

    # Boxplots
    st.subheader("🗂️ Category-wise & Seasonal Trends")
    box1, box2 = st.columns(2)

    with box1:
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=filtered_df, x="Product_Category", y="Quantity_Sold", palette="pastel", ax=ax4)
        ax4.set_title("Quantity by Product Category")
        ax4.tick_params(axis='x', rotation=45)
        st.pyplot(fig4)

    with box2:
        fig5, ax5 = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=filtered_df, x="Season", y="Selling_Price", palette="Set2", ax=ax5)
        ax5.set_title("Selling Price by Season")
        st.pyplot(fig5)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align:center; font-size: 14px;">
    Made by <strong>Maansi Tomer</strong> | 👗 AI Clothing Sales Intelligence | Streamlit Dashboard
    </div>
""", unsafe_allow_html=True)
