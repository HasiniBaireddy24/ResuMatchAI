from data_extractor import normalize_skill # Import the normalize_skill function

def calculate_match_score(resume_skills, job_skills_list):
    """"
    Calculates the match score between resume skills and job description skills.
    Args :
    resume_skills(list) : Skills extracted from the resume.
    job_skills_list(list) : Skills extracted from the JD

    Returns:
    tuple: (match_score, matched_skills, missing_skills)
    """
    
    normalized_resume_skills = {normalize_skill(skill) for skill in resume_skills}
    normalized_job_skills = {normalize_skill(skill) for skill in job_skills_list}

    if not normalized_job_skills:
        return 0.0, [], [] 

    matched_skills = normalized_resume_skills.intersection(normalized_job_skills)
    missing_skills = normalized_job_skills.difference(normalized_resume_skills)

    total_job_skills = len(normalized_job_skills)
    
    match_score = (len(matched_skills) / total_job_skills) * 100
    return round(match_score, 2), list(matched_skills), list(missing_skills)