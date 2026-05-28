import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pypdf  # Library to handle PDF text extraction natively

# Set professional layout configuration
st.set_page_config(page_title="Student Placement Evaluation Hub", layout="wide")

# Custom Clean Dashboard Styling (Dark Premium Theme)
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    .card { background-color: #161b22; border: 1px solid #30363d; padding: 20px; border-radius: 8px; margin-bottom: 15px;}
    .badge { padding: 10px 20px; border-radius: 6px; font-weight: bold; font-size: 1.4rem; display: inline-block; text-align: center;}
    .very-likely { background-color: #238636; color: white; }
    .likely { background-color: #2ea043; color: white; }
    .moderate { background-color: #d29922; color: black; }
    .low { background-color: #f85149; color: white; }
    .very-low { background-color: #da3633; color: white; }
    </style>
    """, unsafe_allow_html=True)

# Helper function to read uploaded PDF files in-memory
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            extracted_page_text = page.extract_text()
            if extracted_page_text:
                text += extracted_page_text + "\n"
        return text.lower()
    except Exception as e:
        st.error(f"Error parsing PDF file: {e}")
        return ""

@st.cache_resource
def load_system_assets():
    with open("student_placement_core.pkl", "rb") as f:
        return pickle.load(f)

try:
    assets = load_system_assets()
    model = assets['model']
    scaler = assets['scaler']
except FileNotFoundError:
    st.error("System assets missing. Please execute `train_model.py` to compile dependencies.")
    st.stop()

st.title("🎓 Student Career Performance & Placement Evaluation Platform")
st.markdown("Analyze baseline academic indices, practical work experience metrics, and portfolio parameters to evaluate job readiness profiles.")
st.markdown("---")

col1, col2 = st.columns([1.1, 1.3])

# ================= LEFT SIDE: USER CONTEXT INPUTS =================
with col1:
    st.subheader("📋 Student Metrics Profile")
    
    # 📂 Upgraded PDF Resume Keyword Scanner Box
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("🔹 **Smart Resume Integration**")
    uploaded_file = st.file_uploader("Upload your resume in PDF format", type=["pdf"])
    
    resume_project_multiplier = 0
    if uploaded_file is not None:
        with st.spinner("Parsing PDF content and extracting text structure..."):
            resume_content = extract_text_from_pdf(uploaded_file)
            
        # Target competencies to scan for
        core_keywords = ["python", "sql", "java", "react", "cloud", "aws", "machine learning", "data", "excel", "git", "developer"]
        matched_skills = [skill.upper() for skill in core_keywords if skill in resume_content]
        
        if len(matched_skills) > 0:
            st.success(f"✓ Detected Industry Skills: {', '.join(matched_skills)}")
            # If they match 4 or more core skills, give them a subtle multiplier boost
            if len(matched_skills) >= 4:
                resume_project_multiplier = 1  
                st.info("💡 *Resume Insights: High technical keyword density detected. Granting a profile quality adjustment multiplier.*")
        else:
            st.warning("⚠️ No technical industry core keywords identified in the PDF content.")
    st.markdown("</div>", unsafe_allow_html=True)

    # 🛠️ The 7 Core Feature Sliders
    iq = st.slider("Cognitive Assessment (IQ Score)", 40, 160, 100)
    prev_sem = st.slider("Previous Semester Grade Metric (GPA)", 0.0, 10.0, 7.0, 0.1)
    cgpa = st.slider("Cumulative Academic CGPA", 0.0, 10.0, 7.2, 0.1)
    
    # Internship Field (Conditional Text Generation)
    internship_opt = st.selectbox("Completed Prior Industry Internships?", ["No", "Yes"])
    internship = 1 if internship_opt == "Yes" else 0
    if internship == 1:
        st.text_area("💼 Summary of Internship Core Contributions", 
                     placeholder="e.g., Collaborated on building frontend components, optimized internal report generation scripts...", height=70)

    # Extra-Curricular Field (Slider + Text Input Combo)
    extra_curr = st.slider("Extra-Curricular Engagement Level (1-10)", 1, 10, 5)
    st.text_input("🎨 Highlight Major Extra-Curricular Engagements", 
                  placeholder="e.g., Active team leader in student technical cell, state-level tennis competitor...")
    
    comm_skills = st.slider("Interview & Communication Competency Rating (1-10)", 1, 10, 6)
    
    base_projects = st.number_input("Technical Core Portfolio Projects Completed", min_value=0, max_value=10, value=2)
    projects = base_projects + resume_project_multiplier

    st.markdown("###")
    trigger_evaluation = st.button("GENERATE CAREER ANALYSIS REPORT", type="primary")

# ================= RIGHT SIDE: SIMPLIFIED RESULTS & RECOMMENDATIONS =================
with col2:
    st.subheader("🔮 Assessment & Placement Likelihood Report")
    
    if trigger_evaluation:
        # Build vector array matching training format exactly
        input_data = np.array([[iq, prev_sem, cgpa, internship, extra_curr, comm_skills, projects]])
        scaled_data = scaler.transform(input_data)
        
        # Pull raw probability from model
        placement_probability = model.predict_proba(scaled_data)[0][1]
        
        # Map raw probabilities directly to clean categories
        if placement_probability >= 0.82:
            status_class = "very-likely"
            status_text = "VERY LIKELY"
        elif placement_probability >= 0.65:
            status_class = "likely"
            status_text = "LIKELY"
        elif placement_probability >= 0.45:
            status_class = "moderate"
            status_text = "MODERATE"
        elif placement_probability >= 0.25:
            status_class = "low"
            status_text = "LOW"
        else:
            status_class = "very-low"
            status_text = "VERY LOW"
            
        # Display the custom colored tier badge
        st.markdown(f"**Current Structural Classification Profile Status:**")
        st.markdown(f"<div class='badge {status_class}'>{status_text}</div>", unsafe_allow_html=True)
        st.markdown(f"### Placement Probability Index: {placement_probability*100:.1f}%")
        st.markdown("---")
        
        # --- PERSONALIZED ACTIONABLE SUGGESTIONS ENGINE ---
        st.markdown("### 📋 Personalized Profile Development Roadmap")
        
        if status_text in ["VERY LIKELY", "LIKELY"]:
            st.success("✨ **Profile Strength Insights:** Your current parameter configurations are highly aligned with target recruitment criteria. To ensure an ideal outcome, maintain your current academic standing and practice Mock Interview dry-runs to polish delivery.")
        else:
            st.warning("⚡ **Profile Growth Required:** Your configuration sits below the preferred recruitment safety threshold. Running an automated diagnostic to build your optimization strategy...")
            
            # Use background simulation loop to find the exact minimum changes to reach "LIKELY" (>=65% probability)
            sim_vector = input_data.copy()
            optimized = False
            
            for d_cgpa in np.arange(0.0, 2.5, 0.1):
                for d_projects in range(0, 4):
                    for d_comm in range(0, 4):
                        
                        sim_vector[0][2] = min(input_data[0][2] + d_cgpa, 10.0)      # Target CGPA
                        sim_vector[0][6] = min(input_data[0][6] + d_projects, 10.0)  # Target Projects
                        sim_vector[0][5] = min(input_data[0][5] + d_comm, 10.0)      # Target Communication
                        
                        test_scaled = scaler.transform(sim_vector)
                        test_prob = model.predict_proba(test_scaled)[0][1]
                        
                        if test_prob >= 0.65:
                            st.markdown("#### **🛠️ Targeted Profile Action Milestones:**")
                            
                            rec_col1, rec_col2, rec_col3 = st.columns(3)
                            with rec_col1:
                                if d_cgpa > 0:
                                    st.metric("Academic Milestone", f"+{d_cgpa:.1f} CGPA", f"Target: {sim_vector[0][2]:.1f}")
                                else:
                                    st.metric("Academic Milestone", "Stable", "CGPA Sufficient")
                            with rec_col2:
                                if d_projects > 0:
                                    st.metric("Portfolio Growth", f"+{d_projects} Project(s)", f"Target: {int(sim_vector[0][6])}")
                                else:
                                    st.metric("Portfolio Growth", "Stable", "Projects Sufficient")
                            with rec_col3:
                                if d_comm > 0:
                                    st.metric("Soft Skills Push", f"+{d_comm} Point(s)", f"Target: {int(sim_vector[0][5])}/10")
                                else:
                                    st.metric("Soft Skills Push", "Stable", "Skills Sufficient")
                            
                            suggestions = ["💡 **Core Recommendations Matrix:**"]
                            if d_cgpa > 0:
                                suggestions.append("- Prioritize core coding labs and regular study blocks next semester to lift your cumulative CGPA average.")
                            if d_projects > 0:
                                suggestions.append(f"- Build {d_projects} full-stack or machine learning portfolios on GitHub to catch recruiter attention.")
                            if d_comm > 0:
                                suggestions.append("- Participate in mock interviews, technical presentations, or student cell discussions to clear the communication cut-offs.")
                            if internship == 0:
                                suggestions.append("- Actively seek micro-internships or freelance projects to convert your practical experience metric from a 0 to a 1.")
                                
                            st.markdown("\n".join(suggestions))
                            optimized = True
                            break
                    if optimized: break
                if optimized: break
    else:
        st.info("Awaiting metric inputs. Configure parameters on the left panel and click 'Generate Career Analysis Report' to display your roadmap.")