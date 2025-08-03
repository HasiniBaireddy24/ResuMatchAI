#  ResuMatchAI

## Streamline Resume Matching with AI

**ResuMatchAI** is a Streamlit-based web application that helps job seekers and recruiters optimize resume-to-job-description (JD) matching. It uses NLP and LLMs to deliver match scores, identify skill gaps, and give AI-powered resume improvement suggestions.

---

##  Key Features

- **Resume-to-JD Matching**: Calculates a match score based on skill comparison.
- **Skill Gap Analysis**: Detects missing skills in resumes.
- **AI-Powered Suggestions**: Recommends resume changes using LLMs.
- **Multi-Resume Support**: Compare multiple candidates.
- **Skill Taxonomy**: Uses a customizable skill alias dictionary.

---

##  Prerequisites

- Python 3.7 or newer
- API Keys for OpenAI and/or Google Gemini

---

##  Installation

Clone the repository:

git clone <your-repository-url>
cd <your-repository-name>

Create and activate a virtual environment:

Using venv:
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

Using Conda:
conda create -n resume-matcher python=3.9
conda activate resume-matcher

Install required packages:

pip install -r requirements.txt
Download the spaCy language model:

python -m spacy download en_core_web_sm
Set your API keys as environment variables:

On macOS/Linux:
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-gemini-key"

On Windows CMD:
set OPENAI_API_KEY="your-openai-key"
set GOOGLE_API_KEY="your-gemini-key"

Usage
After activating the environment, run the following command:

streamlit run app.py

This will start a local development server and open the app in your browser.

File Structure
app.py               # Main Streamlit application
data_extractor.py    # Extracts personal info and skills from resumes
job_parser.py        # Extracts skills from job descriptions
matcher.py           # Match score logic and skill gap identification
skill_aliases.py     # Dictionary for skill normalization
llm_utils.py         # Utility functions for LLM integration
resume_parser.py     # Parses resumes (PDF to text)
 requirements.txt     # Project dependencies
📄 License
This project is licensed under the MIT License.
See the LICENSE file for more details.
