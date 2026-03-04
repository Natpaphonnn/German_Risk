import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Project Story", page_icon="📊", layout="wide")

# --- Load data ---
@st.cache_data
def load_data():
    # Try multiple paths
    for path in ["../data/german_credit_data.csv", "../../german_credit_data.csv", "../german_credit_data.csv", "german_credit_data.csv"]:
        full = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(full):
            df = pd.read_csv(full)
            if df.columns[0].startswith("Unnamed"):
                df = df.drop(columns=[df.columns[0]])
            return df
    return None

df = load_data()

# --- Header ---
st.markdown("# 📊 Project Story")
st.markdown("เล่าให้ฟังทีละขั้นว่าเราทำอะไร ค้นพบอะไร และได้ผลลัพธ์อย่างไร")
st.divider()

# ==========================================
# CHAPTER 1: ข้อมูลคืออะไร
# ==========================================
st.markdown("## 1. ข้อมูลที่ใช้")
st.markdown("""
ชุดข้อมูล **German Credit Dataset** เก็บข้อมูลผู้ขอสินเชื่อจากธนาคารในเยอรมนี จำนวน **1,000 คน**
แต่ละคนมีข้อมูล 9 ด้าน เช่น อายุ, เพศ, อาชีพ, ยอดกู้, ระยะเวลากู้ ฯลฯ
และผลลัพธ์ว่า **"ดี" (good)** หรือ **"เสี่ยง" (bad)**
""")

if df is not None:
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown("#### ตัวอย่างข้อมูล")
        st.dataframe(df.head(8), use_container_width=True, height=320)

    with col2:
        st.markdown("#### สัดส่วน Good vs Bad")
        risk_counts = df["Risk"].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color_discrete_sequence=["#66c2a5", "#fc8d62"],
            hole=0.4,
        )
        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
        fig.update_traces(textinfo="label+percent", textfont_size=14)
        st.plotly_chart(fig, use_container_width=True)

    st.info("💡 **70% เป็น good, 30% เป็น bad** — ข้อมูลไม่สมดุล ต้องจัดการเรื่องนี้ตอนสร้างโมเดล")

st.divider()

# ==========================================
# CHAPTER 2: ค่าที่หายไป
# ==========================================
st.markdown("## 2. ค่าที่หายไป (Missing Values)")
st.markdown("""
ข้อมูลส่วนใหญ่ครบ แต่มี **2 คอลัมน์** ที่ข้อมูลหาย:
""")

col1, col2 = st.columns(2)
with col1:
    st.metric("Saving Accounts", "183 หาย", "18.3%")
with col2:
    st.metric("Checking Account", "394 หาย", "39.4%")

st.markdown("""
เราเติมค่าที่หายด้วย **"unknown"** แทนที่จะลบทิ้ง เพราะ "ไม่มีข้อมูล" อาจหมายถึง **ไม่มีบัญชี**
ซึ่งเป็นข้อมูลที่มีความหมาย — และจริง ๆ แล้ว กลุ่ม unknown มี **bad rate สูงมาก!**
""")

st.divider()

# ==========================================
# CHAPTER 3: สิ่งที่ค้นพบจาก EDA
# ==========================================
st.markdown("## 3. สิ่งที่ค้นพบจากการสำรวจข้อมูล")

if df is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["🏦 Checking Account", "👤 อายุ", "💰 ยอดเงินกู้", "⏱️ ระยะเวลากู้"])

    with tab1:
        st.markdown("#### คนที่ไม่มีข้อมูล Checking Account เสี่ยงที่สุด")
        df_temp = df.copy()
        df_temp["Checking account"] = df_temp["Checking account"].fillna("unknown")
        ct = pd.crosstab(df_temp["Checking account"], df_temp["Risk"], normalize="index").round(3) * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(x=ct.index, y=ct["bad"], name="Bad %", marker_color="#fc8d62"))
        fig.add_trace(go.Bar(x=ct.index, y=ct["good"], name="Good %", marker_color="#66c2a5"))
        fig.update_layout(barmode="stack", height=400, yaxis_title="เปอร์เซ็นต์ (%)",
                          margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)
        st.warning("⚠️ กลุ่ม **unknown** (ไม่มีข้อมูลบัญชี) มี bad rate สูงถึง **88%**")

    with tab2:
        st.markdown("#### คนอายุน้อยเสี่ยงกว่าคนอายุมาก")
        fig = px.box(df, x="Risk", y="Age", color="Risk",
                     color_discrete_sequence=["#66c2a5", "#fc8d62"])
        fig.update_layout(height=400, showlegend=False, margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("อายุ ≤ 25 (Young)", "Bad rate 58%", "สูงสุด", delta_color="inverse")
        col_b.metric("อายุ 26-45 (Adult)", "Bad rate 28%", "ปานกลาง")
        col_c.metric("อายุ 46+ (Senior)", "Bad rate 25%", "ต่ำสุด")

    with tab3:
        st.markdown("#### ยอดเงินกู้สูง → เสี่ยงมากขึ้น")
        fig = px.box(df, x="Risk", y="Credit amount", color="Risk",
                     color_discrete_sequence=["#66c2a5", "#fc8d62"])
        fig.update_layout(height=400, showlegend=False, margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 กลุ่ม **bad** มี median ยอดเงินกู้สูงกว่า good อย่างชัดเจน")

    with tab4:
        st.markdown("#### กู้นานขึ้น → เสี่ยงมากขึ้น")
        fig = px.box(df, x="Risk", y="Duration", color="Risk",
                     color_discrete_sequence=["#66c2a5", "#fc8d62"])
        fig.update_layout(height=400, showlegend=False, margin=dict(t=30, b=30),
                          yaxis_title="Duration (เดือน)")
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 กลุ่ม **bad** มีระยะเวลากู้เฉลี่ยนานกว่า good ประมาณ 6-8 เดือน")

st.divider()

# ==========================================
# CHAPTER 4: Feature Engineering
# ==========================================
st.markdown("## 4. การสร้าง Features ใหม่")
st.markdown("""
นอกจากข้อมูลดิบ เราสร้าง **3 features ใหม่** ที่ช่วยโมเดลเข้าใจข้อมูลดีขึ้น:
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background:#f0f7ff; padding:1.2rem; border-radius:10px; border-left:4px solid #4a90d9;">
        <b>💸 ค่าผ่อน/เดือน</b><br>
        <small>= ยอดเงินกู้ ÷ ระยะเวลา</small><br><br>
        <small style="color:#666;">คนที่ผ่อนน้อยแต่กู้เยอะ → เสี่ยง</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background:#f0fff0; padding:1.2rem; border-radius:10px; border-left:4px solid #66c2a5;">
        <b>👤 กลุ่มอายุ</b><br>
        <small>Young (≤25) / Adult (26-45) / Senior (46+)</small><br><br>
        <small style="color:#666;">Young มี bad rate สูงสุด 58%</small>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background:#fff8f0; padding:1.2rem; border-radius:10px; border-left:4px solid #fc8d62;">
        <b>📊 ภาระหนี้ต่ออายุ</b><br>
        <small>= ยอดเงินกู้ ÷ อายุ</small><br><br>
        <small style="color:#666;">คนอายุน้อยที่กู้เยอะ → เสี่ยงมาก</small>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==========================================
# CHAPTER 5: ผลลัพธ์โมเดล
# ==========================================
st.markdown("## 5. ผลลัพธ์การเทรนโมเดล")
st.markdown("""
เราเทรน **5 โมเดล** แล้วเปรียบเทียบ — ใช้ค่า **ROC-AUC** เป็นตัววัด
(AUC ยิ่งสูงยิ่งดี โดย 1.0 = สมบูรณ์แบบ, 0.5 = สุ่มเดา)
""")

# Model results data
model_data = pd.DataFrame({
    "Model": ["Random Forest", "Logistic Regression", "SVM", "Gradient Boosting", "XGBoost"],
    "AUC": [0.779, 0.771, 0.767, 0.745, 0.750],
    "Accuracy": [0.735, 0.745, 0.775, 0.755, 0.740],
})

fig = px.bar(
    model_data.sort_values("AUC", ascending=True),
    x="AUC", y="Model", orientation="h",
    color="AUC",
    color_continuous_scale=["#fc8d62", "#66c2a5"],
    text="AUC",
)
fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig.update_layout(
    height=350,
    xaxis_range=[0.5, 0.85],
    xaxis_title="ROC-AUC Score",
    yaxis_title="",
    coloraxis_showscale=False,
    margin=dict(t=30, b=30, l=0, r=50),
)
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.success("""
    🏆 **Random Forest ชนะด้วย AUC = 0.779**
    แม้จะเป็นโมเดลพื้นฐาน (ไม่ tune) แต่ให้ผลดีที่สุด
    """)

with col2:
    st.markdown("""
    **เราลองปรับปรุงโมเดลด้วย:**
    - ✅ SMOTE — สร้างข้อมูล bad เพิ่มให้สมดุล
    - ✅ GridSearchCV — หา hyperparameters ที่ดีที่สุด
    - ผลลัพธ์: AUC ใกล้เคียงกัน → โมเดลพื้นฐานก็ดีพอแล้ว
    """)

st.divider()

# ==========================================
# CHAPTER 6: Feature Importance
# ==========================================
st.markdown("## 6. ปัจจัยที่มีผลต่อความเสี่ยงมากที่สุด")
st.markdown("จากการวิเคราะห์ด้วย **SHAP** และ **Feature Importance** พบว่า:")

importance_data = pd.DataFrame({
    "Feature": ["Checking Account", "Credit Amount", "Duration", "Age", "Saving Account",
                 "Purpose", "Monthly Payment", "Credit per Age", "Housing", "Sex", "Job", "Age Group"],
    "Importance": [0.18, 0.15, 0.13, 0.12, 0.10, 0.08, 0.07, 0.06, 0.04, 0.03, 0.02, 0.02],
})

fig = px.bar(
    importance_data.sort_values("Importance", ascending=True),
    x="Importance", y="Feature", orientation="h",
    color="Importance",
    color_continuous_scale=["#e8e8e8", "#4a90d9"],
    text="Importance",
)
fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig.update_layout(
    height=450,
    coloraxis_showscale=False,
    xaxis_title="Importance Score",
    yaxis_title="",
    margin=dict(t=30, b=30, l=0, r=50),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### สรุปง่าย ๆ

| อันดับ | ปัจจัย | ทำไมถึงสำคัญ |
|:---:|---|---|
| 1 | **Checking Account** | ไม่มีบัญชีเดินสะพัด = เสี่ยงมาก |
| 2 | **Credit Amount** | กู้มาก = เสี่ยงมาก |
| 3 | **Duration** | กู้นาน = เสี่ยงมาก |
| 4 | **Age** | อายุน้อย = เสี่ยงกว่า |
| 5 | **Saving Account** | ไม่มีเงินออม = เสี่ยง |
""")

st.divider()

# ==========================================
# CHAPTER 7: Conclusion
# ==========================================
st.markdown("## 7. สรุป")

st.markdown("""
<div style="background: linear-gradient(135deg, #f0f7ff, #f0fff0); padding: 2rem; border-radius: 12px;">

**โมเดลนี้สามารถช่วยธนาคาร:**

- 🎯 **คัดกรองผู้กู้** ที่มีความเสี่ยงสูงออกก่อนอนุมัติ
- 💰 **ลดความเสียหาย** จากหนี้เสีย
- ⚡ **ตัดสินใจเร็วขึ้น** ด้วยข้อมูลแทนการเดา
- 📊 **อธิบายได้** ว่าทำไมถึงตัดสินใจแบบนั้น (ด้วย SHAP)

**ข้อจำกัด:**
- ข้อมูลมีแค่ 1,000 ราย — ในโลกจริงต้องการข้อมูลมากกว่านี้
- AUC 0.779 ถือว่าดี แต่ยังมีช่องว่างให้ปรับปรุง
- ควรทดสอบกับข้อมูลจริงก่อนนำไปใช้งาน

</div>
""", unsafe_allow_html=True)
