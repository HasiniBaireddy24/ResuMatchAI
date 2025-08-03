import spacy
import re # Import regex for more precise matching
from skill_aliases import skill_aliases # Import skill_aliases

# Load the small English model for spaCy
nlp = spacy.load("en_core_web_sm")

def get_skill_taxonomy():
    """
    Returns a comprehensive list of known skills and their aliases.
    This list should be maintained and expanded for best results.
    The function returns a mapping from all skills/aliases (in lowercase)
    to their canonical, correctly capitalized form.
    """
    # Technical Skills / Programming Languages
    technical = [
        "Python", "R", "SQL", "Java", "C++", "C#", "JavaScript", "Go", "Scala", "Bash",
        "Git", "Docker", "Kubernetes", "Linux", "Unix", "Shell Scripting"
    ]

    # Data Tools / Databases / Platforms
    data_tools = [
        "Tableau", "Power BI", "Looker", "QlikView", "SSRS", "Excel", "VBA",
        "AWS", "Azure", "Google Cloud", "GCP", "Snowflake", "Databricks", "Redshift",
        "BigQuery", "S3", "Athena", "Glue", "EC2", "RDS", "DynamoDB", "MongoDB", "Cassandra",
        "Hadoop", "Spark", "Kafka", "Elasticsearch", "Airflow", "dbt", "PostgreSQL",
        "MySQL", "Oracle", "SQL Server"
    ]

    # Data Science / Machine Learning / NLP Concepts & Libraries
    ds_ml_nlp = [
        "Machine Learning", "Deep Learning", "NLP", "Natural Language Processing",
        "Computer Vision", "Statistical Modeling", "Predictive Modeling",
        "Time Series Analysis", "Reinforcement Learning", "Data Mining",
        "Feature Engineering", "Model Deployment", "MLOps", "A/B Testing",
        "Scikit-learn", "TensorFlow", "Keras", "PyTorch", "NumPy", "Pandas",
        "Matplotlib", "Seaborn", "NLTK", "SpaCy", "OpenCV", "XGBoost", "LightGBM"
    ]

    # Methodologies / Concepts
    methodologies = [
        "Data Analysis", "Data Visualization", "Data Wrangling", "ETL", "ELT",
        "Data Warehousing", "Data Governance", "Big Data", "Business Intelligence",
        "BI", "Statistical Analysis", "Hypothesis Testing", "Dashboarding", "Reporting",
        "A/B testing", "Experiment Design", "Cloud Computing", "Version Control",
        "Agile", "Scrum", "Kanban", "DevOps", "CI/CD", "Data Modeling", "Data Architecture",
        "Requirements Gathering", "Stakeholder Management", "Data Storytelling"
    ]

    # Soft Skills
    # Focus on those that are often specifically looked for in JDs.
    soft_skills = [
        "Communication", "Problem Solving", "Critical Thinking", "Teamwork",
        "Collaboration", "Analytical Skills", "Presentation Skills", "Interpersonal Skills"
    ]

    all_skills = (
        technical + data_tools + ds_ml_nlp + methodologies + soft_skills
    )

   
    skill_map = {}
    for skill in all_skills:
        skill_map[skill.lower()] = skill
    
    # Integrate aliases from skill_aliases.py into the taxonomy for extraction
    for alias, canonical_name in skill_aliases.items():
        if alias not in skill_map: 
            skill_map[alias.lower()] = canonical_name 
        
        if canonical_name.lower() not in skill_map:
            skill_map[canonical_name.lower()] = canonical_name

    return skill_map

SKILL_TAXONOMY = get_skill_taxonomy()

def extract_skills_from_jd(jd_text):
    """
    Extracts relevant skills from a job description using a predefined taxonomy
    and NLP techniques for more accurate matching.
    """
    jd_text_lower = jd_text.lower() # Work with lowercase for matching
    doc = nlp(jd_text_lower)
    extracted_skills = set()

    # Strategy 1: Direct phrase matching (for multi-word skills like "machine learning")
    for lower_skill_phrase, original_skill_phrase in SKILL_TAXONOMY.items():
        # Use word boundaries (\b) to ensure whole word match (e.g., 'r' doesn't match 'research')
        if len(lower_skill_phrase) == 1 and lower_skill_phrase.isalpha():
            # For single letters, check if it's a standalone word or followed by non-alphanumeric
            if re.search(r'\b' + re.escape(lower_skill_phrase) + r'(\b|\W)', jd_text_lower):
                extracted_skills.add(original_skill_phrase)
        elif re.search(r'\b' + re.escape(lower_skill_phrase) + r'\b', jd_text_lower):
            extracted_skills.add(original_skill_phrase)

    # Strategy 2: Token-based matching (for single-word skills like "python" or "sql")
    for token in doc:
        if token.text in SKILL_TAXONOMY: # Check if the token (in its original form) is a key in the taxonomy
            extracted_skills.add(SKILL_TAXONOMY[token.text])

    # Strategy 3: Noun chunk analysis (with validation against taxonomy)
    for chunk in doc.noun_chunks:
        chunk_text_lower = chunk.text.lower()
        if chunk_text_lower in SKILL_TAXONOMY:
            extracted_skills.add(SKILL_TAXONOMY[chunk_text_lower])
        else:
            # Check if any significant word in the chunk is a known skill
            for token in chunk:
                generic_words_to_avoid = {"data", "tools", "system", "analytics", "analysis", "learning", "processing", "intelligence", "models"}
                
                if token.text in SKILL_TAXONOMY and token.text not in generic_words_to_avoid:
                    extracted_skills.add(SKILL_TAXONOMY[token.text])

    # Final post-processing for consistent capitalization (using the SKILL_TAXONOMY map values)
    final_skills = set()
    for skill_found in extracted_skills:
        canonical_skill = SKILL_TAXONOMY.get(skill_found.lower(), skill_found)
        final_skills.add(canonical_skill)

    return list(sorted(final_skills))

# Example Usage (for testing this file independently)
if __name__ == '__main__':
    sample_jd = """
    We are seeking a highly motivated Data Analyst with strong Python and SQL skills.
    Proficiency in Tableau and Power BI is a must.
    Experience with AWS cloud services and machine learning concepts (ML) is highly valued.
    The ideal candidate will have excellent communication and analytical skills,
    and a proven track record of creating compelling data visualizations and dashboards.
    Familiarity with large datasets, data wrangling, and basic statistics is essential.
    Knowledge of NLP and other BI tools (e.g., Looker) is a plus.
    Previous internships or academic projects in data analysis are preferred.
    We also require experience in CI/CD pipelines and Agile methodologies.
    """
    extracted = extract_skills_from_jd(sample_jd)
    print("Extracted Skills from JD:")
    for skill in extracted:
        print(f"- {skill}")
