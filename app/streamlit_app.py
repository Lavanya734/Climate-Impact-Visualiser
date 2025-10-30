# app/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.linear_model import LinearRegression

# ------------------------- PAGE CONFIGURATION -------------------------
st.set_page_config(page_title="Climate Impact Visualizer", layout="wide")

# ------------------------- LOAD DATA & MODEL -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_climate_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/trained_model.joblib")

df = load_data()
model = load_model()

# ------------------------- HEADER SECTION -------------------------
st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>🌾 Climate Impact Visualizer</h1>
    <p style='text-align: center; font-size:18px;'>Analyze how <b>average temperature</b> and <b>rainfall</b> influence crop yield across regions.</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ------------------------- SIDEBAR FILTERS -------------------------
st.sidebar.title("Filters & Inputs")
st.sidebar.markdown("Select options to explore the dataset and make predictions.")

area = st.sidebar.selectbox("Select Area", df["Area"].unique())
item = st.sidebar.selectbox("Select Item", df["Item"].unique() if "Item" in df.columns else ["All"])

filtered_df = df[df["Area"] == area]
if "Item" in df.columns and item != "All":
    filtered_df = filtered_df[filtered_df["Item"] == item]

# ------------------------- TABS LAYOUT -------------------------
tab1, tab2, tab3 = st.tabs(["Climate Trends", "Yield Analysis", "Yield Prediction"])

# ------------------------- TAB 1: CLIMATE TRENDS -------------------------
with tab1:
    st.subheader(f"Climate Trends in {area}")
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.line(
            filtered_df, x="Year", y="Avg_rainfall",
            title="Average Rainfall Over Years",
            markers=True, color_discrete_sequence=["#00BFFF"]
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.line(
            filtered_df, x="Year", y="Avg_Temp",
            title="Average Temperature Over Years",
            markers=True, color_discrete_sequence=["#FF6347"]
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
        **Insights:**  
        - Higher rainfall often supports better crop yield, especially in stable climate zones.  
        - Temperature spikes may lead to yield drops or crop stress.  
    """)

# ------------------------- TAB 2: YIELD ANALYSIS -------------------------
with tab2:
    st.subheader(f"Crop Yield Trends in {area}")
    fig3 = px.bar(
        filtered_df, x="Year", y="Crop_Yield(hectares)", color="Crop_Yield(hectares)",
        color_continuous_scale="Greens", title="Yearly Crop Yield Trend"
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    avg_yield = filtered_df["Crop_Yield(hectares)"].mean()
    max_yield = filtered_df["Crop_Yield(hectares)"].max()
    min_yield = filtered_df["Crop_Yield(hectares)"].min()

    st.markdown(f"""
    **Summary for {area}:**
    - Average Yield: **{avg_yield:.2f} hectares**  
    - Highest Yield Recorded: **{max_yield:.2f} hectares**  
    - Lowest Yield Recorded: **{min_yield:.2f} hectares**
    """)

# ------------------------- TAB 3: YIELD PREDICTION -------------------------
# ------------------------- TAB 3: YIELD PREDICTION -------------------------
with tab3:
    st.subheader("Predict Crop Yield Based on Climate Conditions")

    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.number_input("Enter Avg Temperature (°C):", min_value=1.0, max_value=35.0, value=21.0)
    with col2:
        rainfall = st.number_input("Enter Avg Rainfall (mm):", min_value=50.0, max_value=3500.0, value=1100.0)
    with col3:
        pesticides = st.number_input("Enter Pesticides used (in tonnes):", min_value=0.0, max_value=370000.0, value=15000.0)

    # compute avg_yield for interpretation (ensure it's always available)
    if "Crop_Yield(hectares)" in filtered_df.columns and not filtered_df["Crop_Yield(hectares)"].dropna().empty:
        avg_yield = filtered_df["Crop_Yield(hectares)"].mean()
    else:
        avg_yield = None

    if st.button("🔮 Predict Yield"):
        # Prepare input as a DataFrame with the same columns as training
        input_data = pd.DataFrame({
            "Avg_Temp": [temperature],
            "Avg_rainfall": [rainfall],
            "Pesticides(tonnes)": [pesticides],
            "Area": [area],
            "Item": [item]
        })

        # Perform one-hot encoding for 'Area' and 'Item'
        input_encoded = pd.get_dummies(input_data, columns=["Area", "Item"])

        # Align encoded input with model’s training columns (if available)
        model_columns = None
        if hasattr(model, "feature_names_in_"):
            model_columns = list(model.feature_names_in_)
        elif isinstance(getattr(model, "coef_", None), (list, tuple)) and hasattr(model, "feature_names_in_"):
            model_columns = list(model.feature_names_in_)

        if model_columns is not None:
            # Reindex to match training columns, fill missing with 0
            input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

        # Ensure columns are only numeric (avoid accidental object dtypes)
        try:
            input_encoded = input_encoded.astype(float)
        except Exception:
            # if some columns still non-numeric, attempt to convert or drop
            input_encoded = input_encoded.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Predict and show
        try:
            prediction = model.predict(input_encoded)[0]
            st.success(f"Predicted Crop Yield: **{prediction:.2f} tons/ha**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            # Stop further interpretation if prediction failed
            

        # Interpretation compared to average yield (if available)
        if avg_yield is not None:
            # note: avg_yield was in 'hectares' in your dataset; ensure this matches model units
            if prediction > avg_yield:
                st.markdown("🌾 The predicted yield is **above average** — favorable conditions!")
            else:
                st.markdown("⚠️ The predicted yield is **below average** — possible stress on crops.")
        else:
            st.markdown("ℹ️ Average yield for this region/item is not available to compare against.")

# ------------------------- INSIGHTS SECTION -------------------------
st.markdown("---")
st.markdown("""
### Key Takeaways
- Consistent rainfall plays a bigger role than total rainfall.  
- Extremely high temperatures (>32°C) reduce yield productivity.  
- Combining both metrics helps identify top-performing regions.  
""")

# ------------------------- FOOTER -------------------------
st.markdown("""
<div style='text-align: center; margin-top: 30px;'>
    <hr>
    <p>Developed by <b>Team Climate Impact Visualiser</b> — Lavanya, Kimish & Himanshi</p>
    <p style='font-size:14px; color:grey;'>IGDTUW | B.Tech CSE (AI)</p>
</div>
""", unsafe_allow_html=True)
