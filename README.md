ResuMatchAI

Project Overview:

ResuMatchAI is a Streamlit-based web application designed to help job seekers and recruiters optimize the resume-to-job-description matching process. By leveraging natural language processing(NLP) and large language models(LLMs), the tool provides a comprehensive analysis of a candidate's resume against a specific job description, offering insights, a match score, and AI-powered suggestions for improvement.

Key Features:

Resume-to-JD Matching : Analyzes skills extracted from a resume and a job description to calculate a percentage match score.

Skill Gap Analysis : Identifies skills listed in the job description that are missing from the candidate's resume.

AI-Powered Suggestions : Provides actionable, LLM-generated recommendations on how to add or rephrase experience to cover missing skills.

Multi-Resume Support : Recruiters can upload multiple resumes to compare candidates and identify common skills across all participants.

Comprehensive Skill Taxonomy : Uses a robust, customizable dictionary of skill aliases to ensure accurate and consistent skill extraction.

Prerequisites:

Before you begin, ensure you have Python 3.7 or newer installed.

Installation:
1. Clone the repository:
git clone <your-repository-url>
cd <your-repository-name>

2. Create and activate a virtual environment(recommended):
Using venv:
python -m venv venv
source venv/bin/activate  #On Windows, use 'venv\Scripts\activate'

Using Conda:
conda create -n resume-matcher python=3.9
conda activate resume-matcher

3. Install the required packages:
pip install -r requirements.txt

4. Download the necessary spaCy language model:
python -m spacy download en_core_web_sm

API Keys:
This application requires  API keys for the large language models. You will need to set them as environment variables:

Google Gemini:

export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"


OpenAI:

export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

Usage:
To run the application, simply execute the following command from the project's root directory after activating your environment:

streamlit run app.py


This will start a local web server, and the application will open automatically in your browser.

File Structure
app.py: The main Streamlit application file.

data_extractor.py: Functions for extracting personal info and skills from resumes.

job_parser.py: Logic for extracting skills from job descriptions.

matcher.py: Contains the function to calculate the match score and identify skill gaps.

skill_aliases.py: A dictionary of skill names and their common aliases for consistent matching.

llm_utils.py: Utility functions for interacting with LLMs.

resume_parser.py: Handles text extraction from PDF files.

requirements.txt: Lists the project dependencies.

License
This project is licensed under the MIT License. See the LICENSE file for details.