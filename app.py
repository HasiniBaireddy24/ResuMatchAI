import streamlit as st
import os
import io
import pandas as pd
import spacy.cli
from skill_aliases import skill_aliases 
import openai
import google.generativeai as genai

from resume_parser import extract_text_from_pdf 
from data_extractor import extract_email, extract_phone, extract_name, extract_skills, normalize_skill 
from job_parser import extract_skills_from_jd 
from matcher import calculate_match_score 

# -------------------- SpaCy Model Download Function --------------------
def download_spacy_model():
    """
    Checks if the 'en_core_web_sm' model is installed and downloads it if not.
    This is the most reliable way to ensure the model is available on
    Streamlit Cloud, as packages.txt can sometimes be inconsistent.
    """
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        st.info("Downloading spaCy model 'en_core_web_sm'... This may take a moment.")
        try:
            spacy.cli.download("en_core_web_sm")
            st.success("SpaCy model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading spaCy model: {e}")
            st.warning("The app may not function correctly without the spaCy model. Please try again.")

download_spacy_model()

# 🔐 Set your API Keys
# Prioritize environment variables for security
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure OpenAI if key is available
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Configure Google Gemini if key is available
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)




@st.cache_data
def cached_extract_text_from_pdf(file_path):
    """Caches the result of text extraction from a PDF file path."""
    return extract_text_from_pdf(file_path)

@st.cache_data
def cached_extract_skills_from_jd(job_description):
    """Caches the result of skill extraction from a job description string."""
    return extract_skills_from_jd(job_description)

@st.cache_data
def cached_calculate_match_score(resume_skills, jd_skills):
    """Caches the result of the match score calculation."""
    return calculate_match_score(resume_skills, jd_skills)

# -------------------- Function Definitions with Caching --------------------

@st.cache_data
def get_llm_suggestions(resume_text, job_description, llm_choice, openai_api_key_input=None, google_api_key_input=None):
    """
    Generates resume improvement suggestions using the chosen LLM.
    This function is now decorated to cache its result for a given set of inputs.
    """
    try:
        system_prompt = "You are a professional resume coach who helps candidates improve their resumes based on job descriptions."
        user_prompt = f"Resume:\n{resume_text}\n\nJob Description:\n{job_description}\n\nPlease provide improvement suggestions for the resume, based on the job description. Highlight any missing areas and recommend what to add."

        token_limit = 3072

        if llm_choice == "OpenAI (GPT-4o-mini)":
            api_key_to_use = openai_api_key_input if openai_api_key_input else OPENAI_API_KEY
            if not api_key_to_use:
                return "❌ Error: OpenAI API Key is not set for GPT-Powered Suggestions."

            client = openai.OpenAI(api_key=api_key_to_use)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=token_limit
            )
            return response.choices[0].message.content.strip()

        elif llm_choice == "Google (Gemini-Pro)":
            api_key_to_use = google_api_key_input if google_api_key_input else GOOGLE_API_KEY
            if not api_key_to_use:
                return "❌ Error: Google API Key is not set for Gemini-Powered Suggestions."
            
            genai.configure(api_key=api_key_to_use)
            
            model = genai.GenerativeModel('gemini-1.5-flash') 

            response = model.generate_content(
                system_prompt + "\n\n" + user_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=token_limit 
                )
            )
            return response.text.strip()

        else:
            return "❌ Error: Please select a valid LLM for suggestions."

    except Exception as e:
        return f"❌ Error from LLM: {e}"

# --- Streamlit App Layout ---

st.set_page_config(
    page_title="AI Resume Screener",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'About': 'This is an AI-powered resume screening tool to help you match your resume with job descriptions.'
    }
)

# CSS
st.markdown("""
    <style>
        /* General page background */
        .stApp {
            background-color: #f0f2f6; /* Light gray background */
        }
        /* Main content block styling */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            background-color: #ffffff; /* White background for the main content */
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); /* Soft shadow */
        }
        /* Header styling */
        h1, h2, h3 {
            color: #2c3e50; /* Darker blue for headers */
            font-family: 'Segoe UI', sans-serif;
            font-weight: 600;
        }
        /* Body text */
        p, .stMarkdown {
            color: #34495e; /* Slightly lighter body text */
            font-family: 'Segoe UI', sans-serif;
        }
        /* Input fields and buttons */
        .stTextInput, .stFileUploader, .stTextArea {
            border: 1px solid #ced4da; /* Lighter border for inputs */
            border-radius: 8px;
            padding: 8px;
            box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05); /* Inner shadow for inputs */
        }
        /* File Uploader specific styling */
        .stFileUploader > label {
            font-weight: 500;
            color: #2c3e50;
        }
        /* Button styling */
        .stButton>button {
            background-color: #87CEEB; /* Light sky blue */
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
            border: none;
            transition: background-color 0.3s ease;
            box_shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        .stButton>button:hover {
            background-color: #6495ED; /* Cornflower blue (slightly darker for hover) */
        }
        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #28a745; /* Green progress bar */
        }
        /* Success/Info/Warning banners */
        .st.success {
            background-color: #d4edda;
            color: #155724;
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid #c3e6cb;
        }
        .st.info {
            background-color: #d1ecf1;
            color: #0c5460;
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid #bee5eb;
        }
        /* Suggestion Box specific styling */
        .suggestion-box {
            background-color: #e8f5e9; /* Lighter green for suggestions */
            padding: 1.2rem;
            border-radius: 10px;
            margin-top: 1.5rem;
            border-left: 5px solid #4CAF50; /* Green border on left */
            box_shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .suggestion-box ul {
            list-style-type: none; /* Remove default bullet points */
            padding-left: 0;
        }
        .suggestion-box li {
            padding: 0.3rem 0;
            border-bottom: 1px dashed #c8e6c9; /* Dotted line between suggestions */
        }
        .suggestion-box li:last-child {
            border-bottom: none;
        }
        /* Highlighted text within st.markdown */
        .highlight {
            background-color: yellow;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'analyze_button_clicked' not in st.session_state:
    st.session_state.analyze_button_clicked = False
if 'job_description_text' not in st.session_state: 
    st.session_state.job_description_text = ""

if 'llm_model_choice' not in st.session_state:
    st.session_state.llm_model_choice = "OpenAI (GPT-4o-mini)" 

if 'openai_api_key_input_val' not in st.session_state:
    st.session_state.openai_api_key_input_val = ""
if 'google_api_key_input_val' not in st.session_state:
    st.session_state.google_api_key_input_val = ""

if 'uploaded_resume_single_file' not in st.session_state:
    st.session_state.uploaded_resume_single_file = None
if 'uploaded_resumes_multiple_files' not in st.session_state:
    st.session_state.uploaded_resumes_multiple_files = []


# --- Stage 1: Role Selection ---
if st.session_state.user_role is None:
    st.title("📄 AI Resume Screener")
    st.markdown("Match your resume with any job description and get smart, actionable insights! ✨")
    st.markdown("---")
    st.subheader("👋 Welcome! Please select your role:")
    
   
    col_seeker, col_recruiter = st.columns(2)

    with col_seeker:
        if st.button("I'm a Job Seeker / Student", key="btn_job_seeker"):
            st.session_state.user_role = "Job Seeker / Student"
            st.rerun() 

    with col_recruiter:
        if st.button("I'm a Recruiter / Hiring Manager", key="btn_recruiter"):
            st.session_state.user_role = "Recruiter / Hiring Manager"
            st.rerun()

else: 
    st.title("📄 AI Resume Screener")
    st.markdown("---") 
    
    # --- Back button for role selection ---
    if st.button("⬅️ Back to Role Selection", key="back_to_role_selection"):
        st.session_state.user_role = None 
        st.session_state.analyze_button_clicked = False 
        st.session_state.job_description_text = "" 
        st.session_state.llm_model_choice = "OpenAI (GPT-4o-mini)" 
        
        st.session_state.openai_api_key_input_val = ""
        st.session_state.google_api_key_input_val = ""
       
        st.session_state.uploaded_resume_single_file = None
        st.session_state.uploaded_resumes_multiple_files = []
        st.rerun() 
    
    st.caption(f"You selected: **{st.session_state.user_role}**")
    st.markdown("---") 

    # --- Input Section ---
    st.header("Upload Resume(s) & Job Description")
    st.markdown("Upload your resume(s) (PDF only) and paste the job description below.")

    uploaded_resumes_for_processing = []

    if st.session_state.user_role == "Job Seeker / Student":
      
        uploaded_resume_single = st.file_uploader(
            "Upload your resume (PDF)",
            type=["pdf"],
            key="resume_uploader_single" 
        )
       
        if uploaded_resume_single:
            st.session_state.uploaded_resume_single_file = uploaded_resume_single
        elif 'resume_uploader_single' in st.session_state and st.session_state['resume_uploader_single'] is None:
            st.session_state.uploaded_resume_single_file = None

        uploaded_resumes_for_processing = [st.session_state.uploaded_resume_single_file] if st.session_state.uploaded_resume_single_file else []

    else: 
        uploaded_resumes_multiple = st.file_uploader(
            "Upload resumes (PDFs)",
            type=["pdf"],
            accept_multiple_files=True,
            key="resume_uploader_multiple"
        )
        
        if uploaded_resumes_multiple:
            st.session_state.uploaded_resumes_multiple_files = uploaded_resumes_multiple
        
        elif 'resume_uploader_multiple' in st.session_state and not st.session_state['resume_uploader_multiple']:
             st.session_state.uploaded_resumes_multiple_files = []

        uploaded_resumes_for_processing = st.session_state.uploaded_resumes_multiple_files


    # Job Description Text Area 
    job_description = st.text_area(
        "Paste the Job Description here",
        value=st.session_state.job_description_text,
        height=250,
        key="jd_input",
        placeholder="e.g., 'We are looking for a Data Scientist with strong Python, SQL, and Machine Learning skills...'"
    )
    st.session_state.job_description_text = job_description


    openai_api_key_input = None
    google_api_key_input = None
    if st.session_state.user_role == "Job Seeker / Student":
        st.markdown("---")
        
        with st.expander("🔧 AI Suggestions Configuration (Optional)", expanded=False):
            llm_choice_options = ("OpenAI (GPT-4o-mini)", "Google (Gemini-Pro)")
            try:
                current_index = llm_choice_options.index(st.session_state.llm_model_choice)
            except ValueError:
                current_index = 0 
                st.session_state.llm_model_choice = llm_choice_options[0]

            llm_choice = st.radio(
                "Choose your AI model for suggestions:",
                llm_choice_options,
                index=current_index, 
                key="llm_model_choice" 
            )

            if llm_choice == "OpenAI (GPT-4o-mini)":
                if not OPENAI_API_KEY:
                    openai_api_key_input = st.text_input(
                        "Enter your OpenAI API Key (for GPT suggestions)",
                        type="password",
                        key="openai_api_key_input",
                        value=st.session_state.openai_api_key_input_val,
                        on_change=lambda: st.session_state.update(openai_api_key_input_val=st.session_state.openai_api_key_input) 
                    )
                    if not openai_api_key_input:
                        st.info("You can get an OpenAI API key from [platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys). This key is not stored.")
                else:
                    st.success("OpenAI API Key is set (via environment variable).")
            elif llm_choice == "Google (Gemini-Pro)":
                if not GOOGLE_API_KEY:
                    google_api_key_input = st.text_input(
                        "Enter your Google API Key (for Gemini suggestions)",
                        type="password",
                        key="google_api_key_input",
                        value=st.session_state.google_api_key_input_val, 
                        on_change=lambda: st.session_state.update(google_api_key_input_val=st.session_state.google_api_key_input) 
                    )
                    if not google_api_key_input:
                        st.info("You can get a Google API key from [makersuite.google.com/build/api-key](https://makersuite.google.com/build/api-key). This key is not stored.")
                else:
                    st.success("Google API Key is set (via environment variable).")


    analyze_button_label = "Analyze Resume" if st.session_state.user_role == "Job Seeker / Student" else "Analyze Resumes"
    if st.button(analyze_button_label, key="analyze_button_click"):
        st.session_state.analyze_button_clicked = True


    # --- Processing and Results Section ---
    if st.session_state.analyze_button_clicked and uploaded_resumes_for_processing and job_description:
        
        resumes_data = [] 
        jd_skill_set_raw = cached_extract_skills_from_jd(job_description)
        jd_skill_set_normalized = {normalize_skill(s) for s in jd_skill_set_raw}
        
        
        with st.spinner("Processing all resumes... This may take a moment."):
            
            for i, uploaded_resume in enumerate(uploaded_resumes_for_processing):
                temp_resume_path = f"temp_resume_{uploaded_resume.name}.pdf" 
                
                
                st.markdown(f"**Currently processing:** {uploaded_resume.name} ({i+1}/{len(uploaded_resumes_for_processing)})") 

               
                expanded_state = True if (len(uploaded_resumes_for_processing) == 1) or (i == 0 and st.session_state.user_role == "Recruiter / Hiring Manager") else False
                with st.expander(f"📊 Results for {uploaded_resume.name} (Resume {i+1})", expanded=expanded_state):
                    try:
                        with st.spinner(f"🔍 Analyzing {uploaded_resume.name} details..."): 
                            with open(temp_resume_path, "wb") as f:
                                f.write(uploaded_resume.read())

                            
                            resume_text = cached_extract_text_from_pdf(temp_resume_path)
                            
                            name = extract_name(resume_text)
                            email = extract_email(resume_text)
                            phone = extract_phone(resume_text)

                            resume_skills_raw = extract_skills(resume_text, skill_aliases)
                            resume_skills_normalized = {normalize_skill(s) for s in resume_skills_raw}

                            match_score, matched_skills, missing_skills = cached_calculate_match_score(resume_skills_normalized, jd_skill_set_normalized)

                        resumes_data.append({
                            "name": name if name else uploaded_resume.name.replace(".pdf", ""),
                            "email": email,
                            "phone": phone,
                            "extracted_skills": resume_skills_normalized, 
                            "match_score_value": match_score, 
                            "match_score_display": f"{match_score:.2f}%",
                            "matched_jd_skills": matched_skills, 
                            "missing_jd_skills": missing_skills 
                        })
                            
                        st.success(f"✅ Analysis Complete for {uploaded_resume.name}!")

                        # --- Resume Summary Section ---
                        st.markdown("---")
                        st.header("1️⃣ Resume Summary")
                        st.write(f"**Candidate Name:** {name if name else 'Not Found 🤷'}")
                        st.write(f"**Email Address:** {email if email else 'Not Found 📧'}")
                        st.write(f"**Phone Number:** {phone if phone else 'Not Found 📞'}")

                        # --- Match Score & Skills Section ---
                        st.markdown("---")
                        st.header("2️⃣ Resume vs Job Match")

                        st.subheader("Overall Match Score")
                        progress_value = int(match_score) if 0 <= match_score <= 100 else (0 if match_score < 0 else 100)
                        st.progress(progress_value)
                        st.markdown(f"### 🎉 **{match_score:.2f}% Match**")

                        col_matched, col_missing = st.columns(2)

                        with col_matched:
                            st.subheader("🟢 Matched Skills")
                            if matched_skills:
                                st.markdown("You possess the following key skills required by the job:")
                                for skill in sorted(matched_skills):
                                    st.markdown(f"- **{skill}** ✅")
                            else:
                                st.info("No direct skill matches found. Consider reviewing your resume or the job description.")

                        with col_missing:
                            st.subheader("🔴 Missing Skills")
                            if missing_skills:
                                st.markdown("The job description requires these skills that were not clearly found in your resume:")
                                for skill in sorted(missing_skills):
                                    st.markdown(f"- **{skill}** ❌")
                            else:
                                st.success("Great! Your resume appears to cover all key skills mentioned in the job description.")

                        # --- Smart Suggestions Section ---
                        if st.session_state.user_role == "Job Seeker / Student":
                            st.markdown("---")
                            st.header("3️⃣ Smart Suggestions")
                            
                            if missing_skills:
                                can_get_llm_suggestions = False
                                
                                if st.session_state.llm_model_choice == "OpenAI (GPT-4o-mini)" and (OPENAI_API_KEY or st.session_state.openai_api_key_input_val):
                                    can_get_llm_suggestions = True
                                elif st.session_state.llm_model_choice == "Google (Gemini-Pro)" and (GOOGLE_API_KEY or st.session_state.google_api_key_input_val):
                                    can_get_llm_suggestions = True
                                
                                if can_get_llm_suggestions:
                                    st.subheader(f"🤖 {st.session_state.llm_model_choice}-Powered Suggestions")
                                    with st.spinner("💡 Asking your AI resume coach...")
                                        llm_feedback = get_llm_suggestions(
                                            resume_text, 
                                            job_description, 
                                            st.session_state.llm_model_choice, 
                                            st.session_state.openai_api_key_input_val, 
                                            st.session_state.google_api_key_input_val
                                        )
                                        st.markdown(llm_feedback)
                                else:
                                    st.info(f"💡 To get {st.session_state.llm_model_choice}-powered suggestions, please provide your API key in the 'Configuration for AI Suggestions' section above.")
                                    st.markdown("💡 Based on the missing skills, here's how you can enhance your resume:")
                                    ignore_terms = {
                                        'we', 'create', 'understanding', 'our team', 'a strong interest', 'reports',
                                        'responsibilities', 'a data analyst intern', 'exposure', 'knowledge', 'familiarity',
                                        'experience', 'qualifications', 'ongoing projects', 'previous internships',
                                        'academic projects', 'the ideal candidate', 'a plus', 'strong communication',
                                        'collaborate', 'communicate', 'tools', 'summaries', 'team', 'excel',
                                        'data', 'analysis', 'project', 'projects', 'analytical', 'skills', 'data science',
                                        'business', 'intelligence', 'bi', 'other', 'work', 'ability', 'proficient',
                                        'design', 'develop', 'implement', 'manage', 'support', 'technical', 'environment',
                                        'solutions', 'requirements', 'systems', 'platform', 'frameworks', 'concepts',
                                        'understanding', 'strong', 'excellent', 'good', 'proven', 'effective',
                                        'leading', 'related', 'various', 'diverse', 'complex', 'real-world', 'deep',
                                        'key', 'critical', 'fundamental', 'advanced', 'practical', 'hands-on', 'best practices',
                                        'methodologies', 'principles', 'strategies', 'insights', 'actionable', 'optimize',
                                        'improve', 'enhance', 'drive', 'deliver', 'contribute', 'participate', 'collaborate with',
                                        'working with', 'experience in', 'familiarity with', 'knowledge of', 'ability to',
                                        'demonstrated', 'effective', 'efficient', 'reliable', 'scalable', 'secure', 'robust',
                                        'high-performance', 'problem-solving', 'critical thinking', 'decision-making',
                                        'communication', 'interpersonal', 'presentation', 'written', 'verbal', 'written and verbal',
                                        'teamwork', 'collaboration', 'independent', 'self-starter', 'detail-oriented',
                                        'organized', 'time management', 'multitasking', 'adaptability', 'flexibility',
                                        'innovation', 'creativity', 'leadership', 'mentoring', 'coaching', 'training',
                                        'customer-focused', 'client-facing', 'stakeholder management', 'cross-functional',
                                        'cross-organizational', 'global', 'international', 'ethical', 'professional',
                                        'integrity', 'confidentiality', 'compliance', 'regulations', 'governance',
                                        'documentation', 'reporting', 'troubleshooting', 'debugging', 'testing', 'deployment',
                                        'maintenance', 'monitoring', 'tuning', 'migration', 'integration', 'automation',
                                        'scripting', 'configuration', 'administration', 'security', 'privacy', 'risk management',
                                        'audit', 'quality assurance', 'validation', 'verification', 'process improvement',
                                        'continuous improvement', 'agile', 'scrum', 'kanban', 'devops', 'ci/cd',
                                        'version control', 'git', 'jira', 'confluence', 'slack', 'microsoft office',
                                        'google suite', 'presentations', 'spreadsheets', 'word processing', 'email',
                                        'scheduling', 'calendaring', 'virtual meetings', 'remote work', 'distributed teams'
                                    }
                                    ignore_terms_set = set(ignore_terms)

                                    cleaned_suggestions = []
                                    for skill in missing_skills:
                                        
                                        if isinstance(skill, list):
                                            skill_str = " ".join(skill)
                                        else:
                                            skill_str = str(skill)

                                        phrases = [s.strip() for s in skill_str.replace("\n", " ").strip().split(',')]
                                        for phrase in phrases:
                                            phrase = phrase.strip()
                                            is_irrelevant = False
                                            words_in_phrase = phrase.lower().split()
                                            if not words_in_phrase:
                                                continue

                                            if phrase.lower() in ignore_terms_set:
                                                is_irrelevant = True
                                            else:
                                                for word in words_in_phrase:
                                                    if word in ignore_terms_set:
                                                        is_irrelevant = True
                                                        break

                                            if not is_irrelevant and 1 <= len(words_in_phrase) <= 5:
                                                formatted_phrase = []
                                                for word in words_in_phrase:
                                                    if word.lower() in ['sql', 'nlp', 'aws', 'bi', 'ci/cd', 'etl', 'mlops', 'gcp', 'ai']:
                                                        formatted_phrase.append(word.upper())
                                                    elif word.lower() == 'power bi':
                                                        formatted_phrase.append('Power BI')
                                                    elif word.lower() == 'google cloud':
                                                        formatted_phrase.append('Google Cloud')
                                                    elif word.lower() == 'machine learning':
                                                        formatted_phrase.append('Machine Learning')
                                                    elif word.lower() == 'data analysis':
                                                        formatted_phrase.append('Data Analysis')
                                                    elif word.lower() == 'data wrangling':
                                                        formatted_phrase.append('Data Wrangling')
                                                    elif word.lower() == 'analytical skills':
                                                        formatted_phrase.append('Analytical Skills')
                                                    elif word.lower() == 'r':
                                                        formatted_phrase.append('R')
                                                    else:
                                                        formatted_phrase.append(word.capitalize())
                                                cleaned_suggestions.append(" ".join(formatted_phrase))

                                    cleaned_suggestions = list(set(cleaned_suggestions))

                                    if cleaned_suggestions:
                                        st.markdown('<div class="suggestion-box">', unsafe_allow_html=True)
                                        st.markdown("Consider explicitly adding the following skills if they are relevant to your experience and you missed them:")
                                        st.markdown("<ul>", unsafe_allow_html=True)
                                        for skill in sorted(cleaned_suggestions):
                                            st.markdown(f"<li>➡️ **{skill}**</li>", unsafe_allow_html=True)
                                        st.markdown("</ul>", unsafe_allow_html=True)
                                        st.markdown("</div>", unsafe_allow_html=True)
                                    else:
                                        st.success("✅ Your resume appears to cover all key skills from the job description! Well done!")
                            else: 
                                st.success("🎉 Excellent! Your resume fully matches all job requirements. You're good to go!")
                        

                    except Exception as e:
                        st.error(f"An error occurred during analysis of {uploaded_resume.name}: {e}. Please try again or check the format of this resume.")
                        st.warning("Ensure your PDF resume is text-searchable (not just an image).")
                    finally:
                        
                        if os.path.exists(temp_resume_path):
                            os.remove(temp_resume_path)

        # --- Recruiter Dashboard: Comparative Analysis ---
        if st.session_state.user_role == "Recruiter / Hiring Manager" and resumes_data:
            st.markdown("---")
            st.header("3️⃣ Recruiter Dashboard: Comparative Analysis") # Numbered 3 for recruiters now
            st.info("Below is a comparative overview of the uploaded resumes against the Job Description.")

            # Sort resumes by match score
            sorted_resumes_data = sorted(resumes_data, key=lambda x: x['match_score_value'], reverse=True)

            st.subheader("Comparative Overview Table")
            display_data = []
            for resume in sorted_resumes_data:
                display_data.append({
                    "Candidate Name": resume['name'],
                    "Match Score": resume['match_score_display'],
                    "Matched Skills Count": len(resume['matched_jd_skills']),
                    "Missing Skills Count": len(resume['missing_jd_skills']),
                    "Email": resume['email'],
                    "Phone": resume['phone']
                })
            
            if display_data:
                df = pd.DataFrame(display_data, use_container_width=True)
            else:
                st.info("No resume data to display in the comparison table.")

            st.subheader("Cross-Candidate Insights")
            if len(resumes_data) > 0:
                
                common_normalized_skills_across_resumes = set.intersection(
                    *[resume['extracted_skills'] for resume in resumes_data]
                )

                
                common_jd_skills_across_all = jd_skill_set_normalized.intersection(common_normalized_skills_across_resumes)

                if common_jd_skills_across_all:
                    st.markdown(f"**Common Job Description Skills Found Across All Uploaded Resumes:**")
                    st.write(", ".join(sorted(list(common_jd_skills_across_all)))) 
                else:
                    st.info("No common job description skills found across all uploaded resumes.")
                
            else: 
                st.info("Upload more than one resume to see cross-candidate insights.")

    elif st.session_s
