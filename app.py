from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import pdfplumber
import torch
from dotenv import load_dotenv
import os
import tempfile
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import json

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)
MODEL_ID = os.getenv("MODEL_ID")

# Configure base model
base_model_name = os.getenv("BASE_MODEL_NAME")
base_model = SentenceTransformer(base_model_name)

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
    Calculate match score between resume and job role using SentenceTransformer.
    
    Args:
        resume_text (str): Text extracted from resume
        job_role (str): Job role description
    
    Returns:
        float: Match score percentage
    """
    try:
        # 使用SentenceTransformer编码文本
        resume_embedding = base_model.encode(resume_text, convert_to_tensor=True)
        job_embedding = base_model.encode(job_role, convert_to_tensor=True)
        
        # 计算余弦相似度
        similarity_score = torch.nn.functional.cosine_similarity(resume_embedding.unsqueeze(0), job_embedding.unsqueeze(0))
        return float(similarity_score[0]) * 100  # 转换为百分比
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating match score: {str(e)}")

def improve_resume(resume_text, job_role):
    """
    使用vLLM服务生成简历改进建议。
    
    Args:
        resume_text (str): 从简历提取的文本
        job_role (str): 职位描述
    
    Returns:
        str: 改进建议
    """
    try:
        prompt = f"""作为一位专业的简历顾问，请分析以下简历和职位要求，提供3-5条具体的、可操作的建议，以提高简历与职位要求的匹配度。

简历内容：
{resume_text}

目标职位：
{job_role}

请重点关注技能、经验和表述方面的差距，提供清晰具体的改进建议，帮助候选人提高竞争力。建议应该具体且可执行，避免泛泛而谈。请用中文回答。"""

        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "你是一位专业的简历顾问，负责提供具体的、可操作的简历改进建议。请始终使用中文回答。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content if response.choices else "暂无建议。"
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成简历建议时出错：{str(e)}")

@app.post("/match-resume/")
async def match_resume(
    file: UploadFile = File(...),
    job_role: str = Form(...)
):
    """
    处理简历匹配请求的端点
    """
    try:
        # 创建临时文件来存储上传的PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name

        # 提取PDF文本
        resume_text = extract_text_from_pdf(tmp_file_path)
        
        # 删除临时文件
        os.unlink(tmp_file_path)
        
        # 计算匹配分数
        match_score = calculate_match_score(resume_text, job_role)
        
        # 获取改进建议
        suggestions = improve_resume(resume_text, job_role)
        
        return {
            "match_score": f"{match_score:.2f}%",
            "suggestions": suggestions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理简历时出错：{str(e)}")
