from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import google.generativeai as genai
from dotenv import load_dotenv
import os
import tempfile

app = FastAPI()

# Load the fine-tuned PEFT model
base_model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained("saved_model")
base_model = AutoModel.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, "saved_model")
model.eval()

# Load API Key for Gemini AI
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF using pdfplumber.
    
    Args:
        pdf_path (str): Path to the PDF file
    
    Returns:
        str: Extracted text from the PDF
    """
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text.strip() if text else "Unable to extract text from the PDF."
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting PDF text: {str(e)}")

def calculate_match_score(resume_text, job_role):
    """
    Calculate match score between resume and job role using PEFT model.
    
    Args:
        resume_text (str): Text extracted from resume
        job_role (str): Job role description
    
    Returns:
        float: Match score percentage
    """
    try:
        # Encode resume and job role separately
        resume_inputs = tokenizer(resume_text, padding=True, truncation=True, return_tensors="pt")
        job_inputs = tokenizer(job_role, padding=True, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            # Use base model to get embeddings
            resume_outputs = model.base_model(
                input_ids=resume_inputs['input_ids'], 
                attention_mask=resume_inputs['attention_mask']
            )
            job_outputs = model.base_model(
                input_ids=job_inputs['input_ids'], 
                attention_mask=job_inputs['attention_mask']
            )
            
            # Use pooler output for similarity calculation
            resume_embedding = resume_outputs.pooler_output
            job_embedding = job_outputs.pooler_output
        
        # Compute cosine similarity score
        similarity_score = torch.nn.functional.cosine_similarity(resume_embedding, job_embedding)
        return float(similarity_score[0]) * 100  # Convert to percentage
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating match score: {str(e)}")

def improve_resume(resume_text, job_role):
    """
    Generate resume improvement suggestions using Gemini AI.
    
    Args:
        resume_text (str): Text extracted from resume
        job_role (str): Job role description
    
    Returns:
        str: Improvement suggestions
    """
    try:
        prompt = f"""
        Given this resume:
        {resume_text}
        
        And this job role:
        {job_role}
        
        Provide 3-5 specific, actionable suggestions to improve the resume's alignment with the job requirements. 
        Focus on skills, experience, and language that would make the candidate more competitive.
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text if response else "No suggestions available."
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating resume suggestions: {str(e)}")

@app.post("/match-resume/")
async def match_resume(file: UploadFile = File(...), job_role: str = Form(...)):
    """
    API endpoint to match resume with job role.
    
    Args:
        file (UploadFile): Uploaded PDF resume
        job_role (str): Job role description
    
    Returns:
        dict: Match score and improvement suggestions
    """
    # Use tempfile to safely handle file uploads
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        try:
            # Save uploaded file
            temp_file.write(await file.read())
            temp_file.close()
            
            # Extract text from PDF
            resume_text = extract_text_from_pdf(temp_file.name)
            
            # Compute match score
            match_score = calculate_match_score(resume_text, job_role)
            
            # Get resume improvement suggestions
            suggestions = improve_resume(resume_text, job_role)
            
            return {
                "match_score": f"{match_score:.2f}%",
                "suggestions": suggestions
            }
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        finally:
            # Ensure temporary file is deleted
            os.unlink(temp_file.name)
