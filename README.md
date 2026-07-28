# DermaSense

**DermaSense** is an AI-powered, mobile-responsive clinical triage support portal designed to classify skin lesions and streamline dermatology referrals. 

It is built as a lightweight, deployment-ready web application using Streamlit and TensorFlow Lite.

---

## Key Features

- **Lightweight TFLite Inference**: Migrated from a heavy `128MB` Keras format to an optimized `32MB` TensorFlow Lite model (`dermasense_skin_model.tflite`), enabling direct deployment on Streamlit Cloud without large-file storage limits.
- **Three-Class Detection**: Classifies lesion images into **Acne** (Low Risk), **Psoriasis** (Moderate Risk), and **Melanoma** (High Risk) with confidence metrics.
- **Dynamic Healthcare Referrals**: Based on the diagnosed category, the portal dynamically filters medical facilities (specifically in the Jammu region) by specialty, displays appropriate clinical alert urgencies, and provides direct Google Maps navigation links.
- **Interactive Patient Portal**:
  - **Skin Analyzer Quiz**: Offers localized clinical advice based on selected skin types and concerns.
  - **Daily Skin Tracker**: A checklist encouraging patients to manage their daily skin health.
- **In-Memory PDF Generation**: Generates official diagnostic summary reports (with metadata, severity levels, action items, and disclaimer text) dynamically as bytes using `fpdf2`, avoiding disk-writing conflicts.
- **Responsive Premium Theme**: Designed with an elegant, responsive dark-mode layout featuring customized cards, progress indicators, and stacked navigation actions optimized for desktop, tablet, and mobile browsers.

---

## Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) (Python Web App Framework)
- **Deep Learning Inference**: TensorFlow Lite (using `tflite-runtime`)
- **PDF Report Generation**: `fpdf2`
- **Data & Image Processing**: NumPy, Pillow (PIL)

---

## Local Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/DermaSense.git
   cd DermaSense
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.10+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**:
   ```bash
   streamlit run app.py
   ```

---

## Disclaimer
*DermaSense is an automated computer vision screening tool, not a medical device. The classification predictions, risk assessments, and action guidelines are for educational and screening references only. They do NOT constitute medical diagnoses or clinical treatment plans. Patients must consult a qualified dermatologist for official clinical evaluations.*
