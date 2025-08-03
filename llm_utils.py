import openai
import streamlit as st

def get_llm_suggestions(job_description, missing_skills, api_key):
    """
    Generates smart suggestions for resume improvement using an LLM.

    Args:
        job_description (str): The full job description text.
        missing_skills (list): A list of skills identified as missing from the resume.
        api_key (str): OpenAI API key.

    Returns:
        str: A string containing suggestions from the LLM, or an error message.
    """
    if not api_key:
        return "Please provide your OpenAI API key for AI-powered suggestions."
    if not missing_skills:
        return "No missing skills to suggest improvements for."

    openai.api_key = api_key

    missing_skills_str = ", ".join(missing_skills)
    prompt = f"""
    You are an expert resume advisor. Given the following job description and a list of skills missing from a candidate's resume,
    provide concise and actionable suggestions on how the candidate can phrase their existing experience or add new sections
    to highlight these missing skills. Focus on practical resume bullet points or section ideas.

    Job Description:
    {job_description}

    Skills missing from Resume:
    {missing_skills_str}

    Provide suggestions in a clear, bulleted list format. For each missing skill, suggest a way to incorporate it.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Or You can upgrade to "gpt-4" if needed
            messages=[
                {"role": "system", "content": "You are a helpful resume optimization assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except openai.APIError as e:
        return f"OpenAI API Error: {str(e)}. This might be due to rate limits, an invalid API key, or other API issues. Please check your key and try again later."
    except Exception as e:
        return f"An unexpected error occurred with the LLM: {e}"