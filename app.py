# ---------------------------- Streamlit Config ----------------------------
# pyrefly: ignore [missing-import]
import streamlit as st
st.set_page_config(page_title="DermaSense - Skin Disease Detector", layout="wide")

# Inject Custom Responsive Premium Dark-Theme Stylesheet (Google Font, Cards, Progress Bars, Media Queries)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Font & Dark-Mode Background Override */
.stApp {
    font-family: 'Inter', sans-serif;
    background-color: #0e1117;
    color: #fafafa;
}

/* Navigation Header Bar */
.nav-bar {
    background-color: rgba(14, 17, 23, 0.85);
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    padding: 14px 40px;
    margin: -6rem -4rem 2rem -4rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    z-index: 999;
}

.nav-logo {
    color: #4ea8de;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}

.nav-links {
    display: flex;
    gap: 24px;
}

.nav-item {
    color: #b0b0b0;
    font-size: 0.95rem;
    font-weight: 500;
    cursor: default;
}

/* Glassmorphism Containers */
.custom-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(5px);
    -webkit-backdrop-filter: blur(5px);
}

.custom-card-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 16px;
    color: #4ea8de;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}

/* Prediction Highlight Banners */
.prediction-box-acne {
    background: linear-gradient(135deg, #1d3557 0%, #457b9d 100%);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    color: #ffffff;
}

.prediction-box-psoriasis {
    background: linear-gradient(135deg, #e65c00 0%, #f9d423 100%);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    color: #ffffff;
}

.prediction-box-melanoma {
    background: linear-gradient(135deg, #780000 0%, #c1121f 100%);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    color: #ffffff;
}

/* Custom Probability Indicators */
.progress-container {
    margin-bottom: 16px;
}

.progress-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.9rem;
    font-weight: 500;
    margin-bottom: 6px;
    color: #e0e0e0;
}

.progress-bg {
    background-color: rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    height: 8px;
    width: 100%;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Recommended Clinic Item Cards */
.clinic-card {
    background: rgba(255, 255, 255, 0.02);
    border-left: 4px solid #4ea8de;
    padding: 14px 18px;
    margin-bottom: 12px;
    border-radius: 0 8px 8px 0;
    transition: all 0.2s ease-in-out;
}

.clinic-card:hover {
    transform: translateX(6px);
    background: rgba(255, 255, 255, 0.05);
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
}

.clinic-actions {
    display: flex;
    gap: 16px;
    margin-top: 8px;
}

/* Professional Disclaimer Box */
.disclaimer-box {
    background-color: rgba(230, 57, 70, 0.06);
    border: 1px solid rgba(230, 57, 70, 0.2);
    border-radius: 10px;
    padding: 18px;
    margin-top: 30px;
    font-size: 0.85rem;
    line-height: 1.5;
    color: #f17c84;
}

/* Media Queries for Responsive Design on Mobile Devices */
@media (max-width: 768px) {
    .nav-bar {
        padding: 12px 20px;
        margin: -6rem -2rem 1.5rem -2rem;
        flex-direction: column;
        gap: 10px;
    }
    .nav-links {
        gap: 16px;
    }
    .custom-card {
        padding: 16px;
        margin-bottom: 14px;
    }
    .custom-card-title {
        font-size: 1.1rem;
        gap: 8px;
    }
    .prediction-box-acne, .prediction-box-psoriasis, .prediction-box-melanoma {
        padding: 16px;
        border-radius: 10px;
    }
    .clinic-card {
        padding: 12px 14px;
    }
    .clinic-actions {
        flex-direction: column;
        gap: 8px;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------- Imports ----------------------------
import numpy as np
from PIL import Image
from fpdf import FPDF
import datetime
import urllib.parse
import os

# Try importing tflite-runtime first, then standard tensorflow
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        st.error("[ERROR] Model runtime library (tflite-runtime or tensorflow) not found.")

# ---------------------------- Model Loading ----------------------------
@st.cache_resource
def load_tflite_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "dermasense_skin_model.tflite")
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------------------- Class Info ----------------------------
class_names = ["Acne", "Melanoma", "Psoriasis"]
disease_info = {
    "Acne": {
        "description": "Acne is a common skin condition that occurs when hair follicles become clogged with oil and dead skin cells.",
        "risk": "Low"
    },
    "Melanoma": {
        "description": "Melanoma is a serious form of skin cancer that begins in cells known as melanocytes.",
        "risk": "High"
    },
    "Psoriasis": {
        "description": "Psoriasis is a chronic autoimmune condition that causes the rapid buildup of skin cells.",
        "risk": "Moderate"
    }
}

# ---------------------------- Clinics Data with Specialties ----------------------------
jammu_clinics = [
    {
        "name": "Government Medical College (GMC), Jammu",
        "address": "Bakshi Nagar, Jammu, J&K",
        "contact": "+91 94191 93337",
        "website": "https://www.gmcjammu.nic.in/",
        "tags": ["Acne", "Melanoma", "Psoriasis"] # General public hospital
    },
    {
        "name": "AIIMS Jammu – Department of Dermatology",
        "address": "Vijaypur, Jammu",
        "contact": "",
        "website": "https://www.aiimsjammu.edu.in/",
        "tags": ["Melanoma", "Psoriasis"] # Tertiary referral care
    },
    {
        "name": "Dr. Palki Sharma – Body Skin Hair Clinic",
        "address": "100 A, D Block, Gandhi Nagar, Jammu",
        "contact": "+91 6006481742 | +91 191-3545361",
        "website": "https://drpalkisharma.com",
        "tags": ["Acne"] # Private aesthetic dermatologist
    },
    {
        "name": "Dr. Mrinal Gupta’s Skin, Hair & Laser Clinic",
        "address": "Gandhi Nagar, Jammu",
        "contact": "",
        "website": "https://www.practo.com/jammu/clinics/skin-clinics",
        "tags": ["Melanoma", "Psoriasis"] # Advanced skin clinic
    },
    {
        "name": "Dr. Soodan’s Skin Institute",
        "address": "Gandhi Nagar, Jammu",
        "contact": "+91 96224 41333 | 0191-2433232",
        "website": "https://drsoodanskincare.com",
        "tags": ["Psoriasis", "Acne"] # General clinical skin specialist
    },
    {
        "name": "Dermawave Skin, Laser & Hair Transplant Clinic",
        "address": "Gandhi Nagar, Jammu",
        "contact": "",
        "website": "https://www.practo.com/jammu/clinics/skin-clinics",
        "tags": ["Acne", "Psoriasis"]
    },
    {
        "name": "Aastha Skin & Dermato-Cosmetic Centre",
        "address": "Karan Nagar, Jammu",
        "contact": "",
        "website": "https://www.practo.com/jammu/clinics/skin-clinics",
        "tags": ["Psoriasis"]
    },
    {
        "name": "Renew – The Aesthetic Clinic",
        "address": "52 D/C, Gandhi Nagar, Jammu",
        "contact": "+91 88999 62666",
        "website": "https://www.instagram.com/renewtheaestheticclinic/",
        "tags": ["Acne"] # Aesthetic skin clinic
    },
    {
        "name": "Treatwell Skin, Hair & Laser Clinic",
        "address": "Jammu Market",
        "contact": "",
        "website": "https://www.practo.com/jammu/clinics/skin-clinics",
        "tags": ["Acne"]
    },
    {
        "name": "Cosmoweave Skin, Laser & Hair Transplant Clinic",
        "address": "Shaheedi Chowk, Jammu",
        "contact": "",
        "website": "https://www.practo.com/jammu/clinics/skin-clinics",
        "tags": ["Acne"]
    },
    {
        "name": "Dr. Batra’s® Homeopathy Clinic",
        "address": "Shop No 34-B, Ground Floor, North Block, Bahu Plaza, Jammu",
        "contact": "+91 90330 01649",
        "website": "https://clinics.drbatras.com/jammu/skin-doctor-bahu-plaza.php",
        "tags": ["Acne"]
    }
]

# ---------------------------- Navigation Header Bar ----------------------------
st.markdown("""
    <div class="nav-bar">
        <div class="nav-logo">DermaSense</div>
        <div class="nav-links">
            <span class="nav-item">Diagnostics Portal</span>
            <span class="nav-item">Healthcare Referrals</span>
            <span class="nav-item">Support</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------------------------- Main Page Header ----------------------------
st.markdown("""
    <div style="text-align: center; margin-top: 10px; margin-bottom: 30px;">
        <h1 style="font-weight: 700; font-size: 2.20rem; color: #4ea8de; margin-bottom: 6px; letter-spacing: -0.5px;">DermaSense</h1>
        <p style="font-size: 1.05rem; color: #b0b0b0; font-weight: 400; margin-top: 0;">AI-powered skin image classification triage and specialist referral portal</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------- Sidebar Interactive Features ----------------------------
st.sidebar.markdown("""
    <div style="margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.08);">
        <h3 style="color:#4ea8de; font-weight:700; margin-top:0;">DermaSense Portal</h3>
        <p style="font-size:0.88rem; color:#b0b0b0; line-height:1.5;">
            An automated dermatological classification screening system to support clinical referral workflows.
        </p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.subheader("Skin Analyzer Quiz")
skin_type = st.sidebar.selectbox(
    "Select Skin Type",
    ["Select Options...", "Dry Skin", "Oily Skin", "Sensitive Skin", "Normal / Combination"]
)

skin_concern = st.sidebar.selectbox(
    "Primary Skin Concern",
    ["Select Options...", "Acne & breakouts", "Redness & inflammation", "Dry patches / Scaling", "General health screening"]
)

if skin_type != "Select Options..." and skin_concern != "Select Options...":
    advice_map = {
        "Dry Skin": "Ensure daily application of lipid-rich moisturizers. Avoid harsh soaps or hot water that strips natural barrier fats.",
        "Oily Skin": "Use gentle gel cleansers and water-based hydrators. Focus care on oil regulation without dehydrating the dermis.",
        "Sensitive Skin": "Choose scent-free formulas. Avoid chemical peeling agents. Perform strict localized patches testing for new items.",
        "Normal / Combination": "Focus hydration where dry (cheeks) and light sebaceous clearance in oilier sections (T-zone)."
    }
    
    st.sidebar.markdown(f"""
        <div style="background-color: rgba(78, 168, 222, 0.08); border: 1px solid rgba(78, 168, 222, 0.2); border-radius: 8px; padding: 14px; margin-top: 15px; font-size: 0.88rem; color: #4ea8de; line-height: 1.45;">
            <b>Daily Support Recommendation:</b><br>{advice_map[skin_type]}
        </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.subheader("Daily Skin Health Tracker")
tracker_1 = st.sidebar.checkbox("Applied Sunscreen (SPF 30+)")
tracker_2 = st.sidebar.checkbox("Consumed Adequate Water (8+ Cups)")
tracker_3 = st.sidebar.checkbox("Completed Gentle Face Wash")
tracker_4 = st.sidebar.checkbox("Moisturized Skin Barrier")

if tracker_1 and tracker_2 and tracker_3 and tracker_4:
    st.sidebar.markdown("""
        <div style="background-color: rgba(34, 197, 94, 0.08); border: 1px solid rgba(34, 197, 94, 0.2); border-radius: 8px; padding: 12px; margin-top: 15px; font-size: 0.85rem; color: #22c55e; font-weight: 600; text-align: center;">
            Daily Skin Health Goal Achieved!
        </div>
    """, unsafe_allow_html=True)

# ---------------------------- Helper Functions ----------------------------
def preprocess_image(image):
    input_shape = input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]
    image = image.convert("RGB").resize((input_width, input_height))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array, image

def generate_pdf(pred_class, confidence, info):
    pdf = FPDF()
    pdf.add_page()
    
    # Border & Navy Header Band
    pdf.set_draw_color(26, 54, 93)
    pdf.set_fill_color(26, 54, 93)
    pdf.rect(0, 0, 210, 40, "F")
    
    # Header Text
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(190, 15, text="DERMASENSE DIAGNOSTIC SUPPORT REPORT", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("helvetica", "", 10)
    pdf.cell(190, 5, text="Generated via Deep Learning Computer Vision Classification", new_x="LMARGIN", new_y="NEXT", align="C")
    
    # Body Styling Setup
    pdf.set_text_color(30, 30, 30)
    pdf.ln(25)
    
    # Report Metadata Block
    report_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_id = f"DS-{np.random.randint(100000, 999999)}"
    
    pdf.set_font("helvetica", "B", 11)
    pdf.cell(95, 8, text=f"Report Reference: {report_id}")
    pdf.cell(95, 8, text=f"Date of Generation: {report_date}", new_x="LMARGIN", new_y="NEXT", align="R")
    
    # Divider Lines
    pdf.line(10, 58, 200, 58)
    pdf.line(10, 59, 200, 59)
    pdf.ln(8)
    
    # Styled Highlights Panel
    pdf.set_fill_color(245, 247, 250)
    pdf.set_draw_color(220, 224, 230)
    pdf.rect(10, 68, 190, 32, "FD")
    
    pdf.set_y(72)
    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(190, 5, text="AI PREDICTED CLASSIFICATION RESULT", new_x="LMARGIN", new_y="NEXT", align="C")
    
    pdf.set_font("helvetica", "B", 15)
    if pred_class == "Melanoma":
        pdf.set_text_color(211, 47, 47) # Red
    elif pred_class == "Psoriasis":
        pdf.set_text_color(245, 124, 0) # Orange
    else:
        pdf.set_text_color(56, 142, 60) # Green
        
    pdf.cell(190, 10, text=f"{pred_class} - Severity Risk Level: {info['risk']} ({confidence:.2f}% Confidence)", new_x="LMARGIN", new_y="NEXT", align="C")
    
    pdf.set_text_color(30, 30, 30)
    pdf.ln(12)
    
    # Description Block
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(190, 8, text="Clinical Description", new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    pdf.set_font("helvetica", "", 11)
    desc_text = info["description"].encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 7, text=desc_text)
    pdf.ln(6)
    
    # Risk Assessment Block
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(190, 8, text="Risk & Action Plan Guidance", new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    pdf.set_font("helvetica", "", 11)
    risk_notes = {
        "Low": "This classification carries a low risk score. Follow up with routine skincare. Schedule a periodic checkup if you observe any shape, boundary, or color shifts over time.",
        "Moderate": "Moderate clinical risk category. We recommend consulting a medical practitioner to rule out scaling complications, and ensuring skin moisture is properly managed.",
        "High": "High risk category. Biopsy and clinical inspection by a board-certified dermatologist are strongly recommended immediately to rule out serious malignancy. Do not delay medical review."
    }
    risk_desc = f"Assessed Risk Profile: {info['risk']} Risk\n\nRecommended Action Plan: {risk_notes[info['risk']]}"
    risk_desc = risk_desc.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 7, text=risk_desc)
    pdf.ln(15)
    
    # Signature block
    pdf.set_y(222)
    pdf.set_font("helvetica", "I", 10)
    pdf.cell(95, 6, text="Reviewer Signature: _______________________")
    pdf.cell(95, 6, text="Model Engine: DermaSense TFLite v1.0", new_x="LMARGIN", new_y="NEXT", align="R")
    
    # Footer Disclaimer
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, 240, 200, 240)
    
    pdf.set_y(242)
    pdf.set_font("helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    disclaimer = (
        "Medical Triage Disclaimer: This automated report is generated using a computer vision artificial intelligence model. "
        "It is provided for preliminary classification support and educational screening references only. It does NOT "
        "constitute medical advice, official clinical diagnosis, or a healthcare treatment plan. Patients must seek "
        "professional, face-to-face evaluations from a licensed dermatologist or clinical care facility for any official diagnosis."
    )
    disclaimer = disclaimer.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 5, text=disclaimer, align="C")
    
    return pdf.output()

def render_probabilities(prediction, class_names):
    html_content = ""
    for i, cls in enumerate(class_names):
        prob_pct = prediction[0][i] * 100
        # Color palettes based on disease type (vibrant grids for dark mode)
        if cls == "Acne":
            fill_color = "linear-gradient(90deg, #4ea8de 0%, #56cfe1 100%)"
        elif cls == "Melanoma":
            fill_color = "linear-gradient(90deg, #e63946 0%, #ff5c6a 100%)"  # Alert red
        else:
            fill_color = "linear-gradient(90deg, #ffb703 0%, #fb8500 100%)"  # Warning orange
            
        html_content += f"""
        <div class="progress-container">
            <div class="progress-label">
                <span>{cls}</span>
                <span>{prob_pct:.2f}%</span>
            </div>
            <div class="progress-bg">
                <div class="progress-fill" style="width: {prob_pct}%; background: {fill_color};"></div>
            </div>
        </div>
        """
    st.markdown(html_content, unsafe_allow_html=True)

# ---------------------------- Main File Uploader ----------------------------
uploaded_file = st.file_uploader("Drag & drop or upload a skin lesion image", type=["jpg", "jpeg", "png"])

# ---------------------------- Layout & Execution Logic ----------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    img_array, display_image = preprocess_image(image)
    
    # 2-Column Responsive Dashboard Layout
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="custom-card-title">Uploaded Lesion Image</div>', unsafe_allow_html=True)
        st.image(display_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        info = disease_info[predicted_class]
        
        # Style classification header depending on prediction
        if predicted_class == "Melanoma":
            box_class = "prediction-box-melanoma"
        elif predicted_class == "Psoriasis":
            box_class = "prediction-box-psoriasis"
        else:
            box_class = "prediction-box-acne"
            
        st.markdown(f"""
            <div class="{box_class}">
                <h3 style="color:white; margin:0; font-size:1.35rem; font-weight:700;">Classification: <b>{predicted_class}</b></h3>
                <p style="color:rgba(255,255,255,0.9); margin:8px 0 0 0; font-size:1.05rem; font-weight:500;">
                    Confidence Score: <b>{confidence:.2f}%</b>
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Class probabilities breakdown
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="custom-card-title">Classification Probability Breakdown</div>', unsafe_allow_html=True)
        render_probabilities(prediction, class_names)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Clinical Details card
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="custom-card-title">Diagnostic Summary</div>', unsafe_allow_html=True)
        
        risk_color = '#2A9D8F' if info['risk']=='Low' else '#F4A261' if info['risk']=='Moderate' else '#E76F51'
        st.markdown(f"""
            <p style="margin-bottom:12px; line-height:1.5; color:#e0e0e0;"><b>Clinical Description:</b> {info['description']}</p>
            <p style="margin-bottom:16px; color:#e0e0e0;"><b>Severity Risk Assessment:</b> 
                <span style="background-color: {risk_color}; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size:0.85rem; letter-spacing:0.5px;">
                    {info['risk']} Risk
                </span>
            </p>
        """, unsafe_allow_html=True)
        
        # In-memory PDF report generation and download
        pdf_bytes = generate_pdf(predicted_class, confidence, info)
        st.download_button(
            label="Download Diagnostic Report (PDF)",
            data=bytes(pdf_bytes),
            file_name=f"DermaSense_Report_{predicted_class}.pdf",
            mime="application/pdf"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Dynamic Referrals based on Predicted Disease Severity
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    if info["risk"] == "High":
        alert_badge = '<span style="background-color:#E53E3E; color:white; padding:3px 8px; border-radius:4px; font-weight:600; font-size:0.75rem; letter-spacing:0.5px;">URGENT CLINICAL REFERRAL</span>'
        referral_msg = (
            "Because this classification indicates a **High Risk** condition, an immediate clinical "
            "consultation and dermoscopy/biopsy are strongly recommended. Below are tertiary medical centers "
            "and oncology-focused specialists in the Jammu region."
        )
    elif info["risk"] == "Moderate":
        alert_badge = '<span style="background-color:#DD6B20; color:white; padding:3px 8px; border-radius:4px; font-weight:600; font-size:0.75rem; letter-spacing:0.5px;">CLINICAL CONSULTATION ADVISORY</span>'
        referral_msg = (
            "For **Moderate Risk** conditions, we recommend scheduling an appointment with a general clinical "
            "dermatologist to establish a diagnosis and management plan. Below are specialized clinics and medical institutes "
            "in the Jammu region."
        )
    else:
        alert_badge = '<span style="background-color:#3182CE; color:white; padding:3px 8px; border-radius:4px; font-weight:600; font-size:0.75rem; letter-spacing:0.5px;">GENERAL DERMATOLOGY REFERRALS</span>'
        referral_msg = (
            "For **Low Risk** conditions such as Acne, you can explore general skincare centers, "
            "cosmetic skin consultants, or private skin institutes. Below are suitable providers in Jammu."
        )

    st.markdown(f'<div class="custom-card-title">Recommended Care Providers &nbsp; {alert_badge}</div>', unsafe_allow_html=True)
    st.markdown(f"<p style='color:#b0b0b0; margin-bottom:18px; font-size:0.95rem; line-height:1.5;'>{referral_msg}</p>", unsafe_allow_html=True)
    
    # Filter clinics dynamically
    filtered_clinics = [c for c in jammu_clinics if predicted_class in c["tags"]]
    
    # Display in a clean structured list
    for clinic in filtered_clinics:
        # Create Google Maps URL
        query = f"{clinic['name']}, {clinic['address']}"
        maps_url = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(query)}"
        
        contact_html = f'<p style="margin:0 0 8px 0; font-size:0.9rem; color:#b0b0b0;">Call: {clinic["contact"]}</p>' if clinic["contact"] else ""
        
        clinic_html = (
            f'<div class="clinic-card">'
            f'<h4 style="margin:0 0 6px 0; color:#fafafa; font-size:1.05rem; font-weight:600;">{clinic["name"]}</h4>'
            f'<p style="margin:0 0 4px 0; font-size:0.9rem; color:#b0b0b0;">Address: {clinic["address"]}</p>'
            f'{contact_html}'
            f'<div class="clinic-actions">'
            f'<a href="{clinic["website"]}" target="_blank" style="font-size:0.85rem; color:#4ea8de; text-decoration:none; font-weight:600;">View Website &rarr;</a>'
            f'<a href="{maps_url}" target="_blank" style="font-size:0.85rem; color:#56cfe1; text-decoration:none; font-weight:600;">Navigate on Google Maps &rarr;</a>'
            f'</div>'
            f'</div>'
        )
        st.markdown(clinic_html, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

    # Global warning disclaimer banner
    st.markdown("""
        <div class="disclaimer-box">
            <b>IMPORTANT MEDICAL NOTICE:</b> DermaSense is an automated computer vision screening tool, not a medical device. 
            The classification predictions, risk assessments, and action guidelines are for educational and screening references only. 
            They do NOT constitute medical diagnoses, clinical evaluations, or physician consultations. Patients must consult a 
            qualified, licensed dermatologist or seek care in a medical clinic for official diagnostics.
        </div>
    """, unsafe_allow_html=True)

# ---------------------------- Full Width Footer ----------------------------
if not uploaded_file:
    # Add spacing to push the footer cleanly to the bottom when no image is uploaded
    st.markdown('<div style="margin-top: 32vh;"></div>', unsafe_allow_html=True)

st.markdown("""
    <div style="background-color: rgba(14, 17, 23, 0.85); padding: 30px 40px; margin: 4rem -4rem -10rem -4rem; text-align: center; color: #E2E8F0; font-size: 0.9rem; box-shadow: 0 -2px 10px rgba(0,0,0,0.2); border-top: 1px solid rgba(255,255,255,0.08); backdrop-filter: blur(10px);">
        <div style="font-weight: 600; margin-bottom: 8px;">DermaSense Clinical Triage & Support Portal</div>
        <div style="font-size: 0.8rem; color: #A0AEC0; margin-bottom: 12px;">Version 1.0.0 &bull; Secure SSL Encryption &bull; HIPAA Compliant Storage Support</div>
        <div style="font-size: 0.8rem; color: #A0AEC0;">
            For clinical inquiries or support: <a href="mailto:support@dermasense.org" style="color: #63B3ED; text-decoration: none;">support@dermasense.org</a>
        </div>
    </div>
""", unsafe_allow_html=True)
