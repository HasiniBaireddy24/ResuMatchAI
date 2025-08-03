import re
import spacy

# Load spaCy model globally to avoid reloading on each function call
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None 

from skill_aliases import skill_aliases

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else None

def extract_phone(text):
    match = re.search(r'(\+91[-\s]?)?\d{10}', text)
    return match.group(0) if match else None

def extract_name(text):
    lines = text.split('\n')
    for line in lines[:5]: 
        line = line.strip()
        if line and re.match(r'^[A-Za-z\s]+$', line) and len(line.split()) >= 2:
            if line.isupper() and len(line) > 5 and ' ' not in line:
                continue
            return line
    return None

def normalize_skill(skill_name):
    """
    Normalizes a skill name using a predefined dictionary of aliases.
    Converts to lowercase, removes non-alphanumeric characters, and then checks aliases.
    Returns the standardized capitalized form if found, otherwise capitalizes
    each word of the cleaned skill name.
    """
    if not isinstance(skill_name, str):
        return skill_name 

    cleaned_skill = re.sub(r'[^a-z0-9\s]', '', skill_name.lower().strip())

    if cleaned_skill in skill_aliases:
        return skill_aliases[cleaned_skill]
    
    return " ".join([word.capitalize() for word in cleaned_skill.split()])

def extract_skills(resume_text, job_skill_set):
  
    if nlp is None: 
        print("SpaCy model not loaded. Skill extraction might be limited.")
        return []

    resume_text_lower = resume_text.lower()
    found_skills_in_resume = set()

    normalized_job_skills_set = {normalize_skill(s) for s in job_skill_set}

    all_skill_patterns = []
    for standard_skill in normalized_job_skills_set:
        all_skill_patterns.append(re.escape(standard_skill.lower()))
    
        for alias_lower, mapped_skill in skill_aliases.items():
            if mapped_skill == standard_skill:
                all_skill_patterns.append(re.escape(alias_lower))
                
    unique_patterns = sorted(list(set(all_skill_patterns)), key=len, reverse=True)
    
  
    combined_pattern = r'\b(' + '|'.join(unique_patterns) + r')\b' if unique_patterns else ""

    
    if combined_pattern:
       
        matches = re.findall(combined_pattern, resume_text_lower)
        for match_str in matches:
            
            normalized_found_skill = normalize_skill(match_str)
           
            if normalized_found_skill in normalized_job_skills_set:
                found_skills_in_resume.add(normalized_found_skill)

    doc = nlp(resume_text_lower)

    for chunk in doc.noun_chunks:
        normalized_chunk = normalize_skill(chunk.text)
        if normalized_chunk in normalized_job_skills_set:
            found_skills_in_resume.add(normalized_chunk)
            continue

        for token in chunk:
            normalized_token = normalize_skill(token.text)

            if normalized_token in normalized_job_skills_set and \
               normalized_token.lower() not in ["data", "tools", "system", "analytics", "engineering", "modeling", "science", "learning", "skills"]: # Added "skills" to common exclusions
                found_skills_in_resume.add(normalized_token)

    return list(sorted(found_skills_in_resume))