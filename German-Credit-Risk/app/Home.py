import streamlit as st

st.set_page_config(
    page_title="German Credit Risk",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for minimal design ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a1a;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #888;
        margin-top: 0.3rem;
    }
    .stButton > button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown('<p class="main-title">German Credit Risk Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">วิเคราะห์ความเสี่ยงสินเชื่อจากข้อมูลผู้กู้ เพื่อช่วยธนาคารตัดสินใจอนุมัติสินเชื่อ</p>', unsafe_allow_html=True)

st.divider()

# --- Key Metrics ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">1,000</div>
        <div class="metric-label">ข้อมูลผู้กู้</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">12</div>
        <div class="metric-label">Features</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">5</div>
        <div class="metric-label">ML Models</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">0.779</div>
        <div class="metric-label">Best AUC Score</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# --- Project Summary ---
st.markdown("### เกี่ยวกับโปรเจกต์นี้")

st.markdown("""
ธนาคารต้องตัดสินใจทุกวันว่าจะ **อนุมัติสินเชื่อ** ให้ลูกค้าหรือไม่
ถ้าอนุมัติให้คนที่มีความเสี่ยงสูง → ธนาคารอาจ **สูญเสียเงิน**
ถ้าปฏิเสธคนที่ดี → ธนาคาร **เสียโอกาส** ในการทำกำไร

โปรเจกต์นี้ใช้ **Machine Learning** วิเคราะห์ข้อมูลผู้กู้ 1,000 ราย
เพื่อสร้างโมเดลที่ช่วยทำนายว่าผู้กู้คนไหน **เสี่ยง** หรือ **ไม่เสี่ยง**
""")

st.markdown("")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 📊 สิ่งที่ทำ")
    st.markdown("""
    1. **สำรวจข้อมูล** — ดูรูปแบบ, ค่าที่หายไป, ความสัมพันธ์
    2. **สร้าง Features ใหม่** — คำนวณค่าผ่อน/เดือน, จัดกลุ่มอายุ
    3. **เทรน 5 โมเดล** — เปรียบเทียบหาตัวที่ดีที่สุด
    4. **อธิบายผล** — ใช้ SHAP บอกว่าทำไมโมเดลถึงตัดสินใจแบบนั้น
    """)

with col_b:
    st.markdown("### 💡 สิ่งที่ค้นพบ")
    st.markdown("""
    - คนที่ **ไม่มีบัญชีเดินสะพัด** เสี่ยงมากที่สุด (bad rate 88%)
    - คน **อายุน้อย (≤25)** มีความเสี่ยงสูงกว่าค่าเฉลี่ย
    - **ยอดกู้สูง + ระยะเวลานาน** = ยิ่งเสี่ยง
    - โมเดล **Random Forest** ทำนายได้ดีที่สุด (AUC 0.779)
    """)

st.divider()

st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.85rem; padding: 1rem 0;">
    👈 เลือกหน้าจาก Sidebar — <b>Project Story</b> ดูรายละเอียด | <b>Predict</b> ทดลองทำนาย
</div>
""", unsafe_allow_html=True)
