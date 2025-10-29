# app/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.linear_model import LinearRegression

# ------------------------- PAGE CONFIGURATION -------------------------
st.set_page_config(page_title="Climate Impact Visualizer 🌾", layout="wide")

# ------------------------- LOAD DATA & MODEL -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/final_crop_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")

df = load_data()
model = load_model()

# ------------------------- HEADER SECTION -------------------------
st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>🌾 Climate Impact Visualizer</h1>
    <p style='text-align: center; font-size:18px;'>Analyze how <b>temperature</b> and <b>rainfall</b> influence crop yield across regions.</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ------------------------- SIDEBAR FILTERS -------------------------
st.sidebar.title("📊 Filters & Inputs")
st.sidebar.markdown("Select options to explore the dataset and make predictions.")

state = st.sidebar.selectbox("Select State", df["State"].unique())
crop = st.sidebar.selectbox("Select Crop", df["Crop"].unique() if "Crop" in df.columns else ["All"])

filtered_df = df[(df["State"] == state)]
if "Crop" in df.columns and crop != "All":
    filtered_df = filtered_df[filtered_df["Crop"] == crop]

# ------------------------- TABS LAYOUT -------------------------
tab1, tab2, tab3 = st.tabs(["📈 Climate Trends", "🌾 Yield Analysis", "🔮 Yield Prediction"])

# ------------------------- TAB 1: CLIMATE TRENDS -------------------------
with tab1:
    st.subheader(f"Climate Trends in {state}")
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.line(filtered_df, x="Year", y="Rainfall", title="Rainfall Trend Over Years", 
                       markers=True, color_discrete_sequence=["#00BFFF"])
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.line(filtered_df, x="Year", y="Temperature", title="Temperature Trend Over Years",
                       markers=True, color_discrete_sequence=["#FF6347"])
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    🌦️ **Insights:**  
    - Higher rainfall usually supports better crop yield, especially in consistent weather zones.  
    - Sudden temperature spikes often correspond with yield drops.  
    """)

# ------------------------- TAB 2: YIELD ANALYSIS -------------------------
with tab2:
    st.subheader(f"Crop Yield Trends in {state}")
    fig3 = px.bar(filtered_df, x="Year", y="Yield", color="Yield",
                  color_continuous_scale="Greens", title="Yearly Crop Yield Trend")
    st.plotly_chart(fig3, use_container_width=True)
    
    avg_yield = filtered_df["Yield"].mean()
    max_yield = filtered_df["Yield"].max()
    min_yield = filtered_df["Yield"].min()

    st.markdown(f"""
    **Summary for {state}:**
    - 🌱 Average Yield: **{avg_yield:.2f} tons/ha**  
    - 🏆 Highest Yield Recorded: **{max_yield:.2f} tons/ha**  
    - ⚠️ Lowest Yield Recorded: **{min_yield:.2f} tons/ha**
    """)

# ------------------------- TAB 3: YIELD PREDICTION -------------------------
with tab3:
    st.subheader("Predict Crop Yield Based on Rainfall & Temperature")

    col1, col2 = st.columns(2)
    with col1:
        rainfall = st.number_input("Enter Rainfall (mm):", min_value=0.0, max_value=500.0, value=200.0)
    with col2:
        temperature = st.number_input("Enter Temperature (°C):", min_value=0.0, max_value=50.0, value=28.0)

    if st.button("🔮 Predict Yield"):
        prediction = model.predict([[temperature, rainfall]])[0]
        st.success(f"Predicted Crop Yield: **{prediction:.2f} tons/ha**")

        if prediction > avg_yield:
            st.markdown("✅ The predicted yield is **above average** — favorable climatic conditions!")
        else:
            st.markdown("⚠️ The predicted yield is **below average** — possible stress on crops.")

# ------------------------- INSIGHTS SECTION -------------------------
st.markdown("---")
st.markdown("""
### 💡 Key Takeaways
- 🌧️ Rainfall consistency is more critical than total rainfall for stable yields.  
- ☀️ High temperatures (>32°C) often reduce yield productivity.  
- 📊 Combining both metrics helps identify the most productive regions.  
""")

# ------------------------- FOOTER -------------------------
st.markdown("""
<div style='text-align: center; margin-top: 30px;'>
    <hr>
    <p>Developed by <b>Team Climate Impact Visualiser</b> — Lavanya 🌸 & Kimish 🌼</p>
    <p style='font-size:14px; color:grey;'>IGDTUW | B.Tech CSE (AI)</p>
</div>
""", unsafe_allow_html=True)
