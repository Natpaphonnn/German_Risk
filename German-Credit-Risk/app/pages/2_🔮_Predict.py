import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Predict Risk", page_icon="🔮", layout="wide")

# --- Load model ---
@st.cache_resource
def load_model():
    for path in [
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "best_model.pkl"),
        os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl"),
    ]:
        if os.path.exists(path):
            return joblib.load(path)
    return None

model_data = load_model()

# --- Header ---
st.markdown("# 🔮 ทำนายความเสี่ยงสินเชื่อ")
st.markdown("กรอกข้อมูลผู้ขอสินเชื่อ แล้วกดทำนาย — ระบบจะบอกว่า **เสี่ยง** หรือ **ไม่เสี่ยง**")
st.divider()

if model_data is None:
    st.error("ไม่พบไฟล์โมเดล กรุณา run notebook เพื่อสร้าง best_model.pkl ก่อน")
    st.stop()

# --- Input Form ---
st.markdown("### กรอกข้อมูลผู้ขอสินเชื่อ")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("อายุ (ปี)", min_value=18, max_value=100, value=30, step=1)
    sex = st.selectbox("เพศ", ["male", "female"])
    job = st.selectbox("ระดับงาน", [0, 1, 2, 3],
                       format_func=lambda x: {
                           0: "0 - ไม่มีทักษะ/ไม่มีถิ่นที่อยู่",
                           1: "1 - ไม่มีทักษะ/มีถิ่นที่อยู่",
                           2: "2 - มีทักษะ",
                           3: "3 - ทักษะสูง/ผู้บริหาร"
                       }[x])

with col2:
    credit_amount = st.number_input("ยอดเงินกู้ (DM)", min_value=100, max_value=50000, value=3000, step=100)
    duration = st.number_input("ระยะเวลากู้ (เดือน)", min_value=1, max_value=72, value=12, step=1)
    purpose = st.selectbox("วัตถุประสงค์", [
        "car", "radio/TV", "furniture/equipment", "business",
        "education", "repairs", "domestic appliances", "vacation/others"
    ])

with col3:
    housing = st.selectbox("ที่อยู่อาศัย", ["own", "rent", "free"])
    saving = st.selectbox("บัญชีออมทรัพย์", ["little", "moderate", "quite rich", "rich", "unknown"])
    checking = st.selectbox("บัญชีเดินสะพัด", ["little", "moderate", "rich", "unknown"])

st.divider()

# --- Predict ---
if st.button("🔮 ทำนายความเสี่ยง", type="primary", use_container_width=True):

    model = model_data["model"]
    scaler = model_data["scaler"]
    encoders = model_data["encoders"]
    feature_cols = model_data["feature_cols"]

    # Calculate engineered features
    monthly_payment = credit_amount / duration
    credit_per_age = credit_amount / age

    if age <= 25:
        age_group = "Young"
    elif age <= 45:
        age_group = "Adult"
    else:
        age_group = "Senior"

    # Encode categorical
    sex_encoded = encoders["Sex"].transform([sex])[0]
    housing_encoded = encoders["Housing"].transform([housing])[0]
    saving_encoded = encoders["Saving accounts"].transform([saving])[0]
    checking_encoded = encoders["Checking account"].transform([checking])[0]
    purpose_encoded = encoders["Purpose"].transform([purpose])[0]
    age_group_encoded = encoders["Age Group"].transform([age_group])[0]

    # Build feature vector
    features = pd.DataFrame([[
        age, job, credit_amount, duration,
        monthly_payment, credit_per_age,
        sex_encoded, housing_encoded,
        saving_encoded, checking_encoded,
        purpose_encoded, age_group_encoded,
    ]], columns=feature_cols)

    # Scale
    features_scaled = pd.DataFrame(
        scaler.transform(features),
        columns=feature_cols
    )

    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    prob_good = probability[0]
    prob_bad = probability[1]

    # --- Result ---
    st.markdown("")

    if prediction == 0:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #d4edda, #c3e6cb); padding: 2rem; border-radius: 12px; text-align: center;">
            <h1 style="color: #155724; margin: 0;">✅ ความเสี่ยงต่ำ (Good)</h1>
            <p style="font-size: 1.2rem; color: #155724; margin-top: 0.5rem;">
                ความมั่นใจ: <b>{prob_good*100:.1f}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f8d7da, #f5c6cb); padding: 2rem; border-radius: 12px; text-align: center;">
            <h1 style="color: #721c24; margin: 0;">⚠️ ความเสี่ยงสูง (Bad)</h1>
            <p style="font-size: 1.2rem; color: #721c24; margin-top: 0.5rem;">
                ความมั่นใจ: <b>{prob_bad*100:.1f}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Probability breakdown
    col1, col2 = st.columns(2)
    with col1:
        st.metric("โอกาสเป็น Good", f"{prob_good*100:.1f}%")
    with col2:
        st.metric("โอกาสเป็น Bad", f"{prob_bad*100:.1f}%")

    # Input summary
    with st.expander("📋 ข้อมูลที่กรอก"):
        summary = pd.DataFrame({
            "ข้อมูล": ["อายุ", "เพศ", "ระดับงาน", "ยอดเงินกู้", "ระยะเวลา",
                       "วัตถุประสงค์", "ที่อยู่", "บัญชีออมทรัพย์", "บัญชีเดินสะพัด",
                       "ค่าผ่อน/เดือน", "กลุ่มอายุ", "ภาระหนี้/อายุ"],
            "ค่า": [f"{age} ปี", sex, f"Level {job}", f"{credit_amount:,} DM", f"{duration} เดือน",
                    purpose, housing, saving, checking,
                    f"{monthly_payment:,.0f} DM", age_group, f"{credit_per_age:,.0f}"],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

# --- Tips ---
st.markdown("")
st.divider()
st.markdown("### 💡 ปัจจัยที่มีผลต่อการทำนายมากที่สุด")
st.markdown("""
1. **บัญชีเดินสะพัด** — ถ้าเลือก "unknown" จะเสี่ยงสูงมาก
2. **ยอดเงินกู้** — ยิ่งสูงยิ่งเสี่ยง
3. **ระยะเวลากู้** — ยิ่งนานยิ่งเสี่ยง
4. **อายุ** — คนอายุน้อยเสี่ยงกว่า

> ลองเปลี่ยนค่าด้านบนแล้วกด **ทำนาย** ใหม่ เพื่อดูว่าแต่ละปัจจัยมีผลอย่างไร
""")
