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
from pathlib import Path
import shutil
from datetime import datetime
import uuid

# Load environment variables
load_dotenv()

app = FastAPI()

# 定义上传目录
UPLOAD_DIR = Path("uploads")
TEMP_DIR = UPLOAD_DIR / "temp"
PROCESSED_DIR = UPLOAD_DIR / "processed"

# 确保目录存在
TEMP_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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

def save_upload_file(upload_file: UploadFile) -> Path:
    """
    保存上传的文件到临时目录
    
    Args:
        upload_file (UploadFile): 上传的文件
    
    Returns:
        Path: 保存的文件路径
    """
    # 生成唯一文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{unique_id}_{upload_file.filename}"
    file_path = TEMP_DIR / filename
    
    try:
        # 保存文件
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return file_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败：{str(e)}")

def cleanup_old_files(directory: Path, max_age_hours: int = 24):
    """
    清理指定目录中的旧文件
    
    Args:
        directory (Path): 要清理的目录
        max_age_hours (int): 文件保留的最大小时数
    """
    current_time = datetime.now()
    for file_path in directory.glob("*"):
        if file_path.is_file():
            file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_age.total_seconds() > max_age_hours * 3600:
                file_path.unlink()

@app.post("/match-resume/")
async def match_resume(
    file: UploadFile = File(...),
    job_role: str = Form(...)
):
    """
    处理简历匹配请求的端点
    """
    try:
        # 清理旧文件
        cleanup_old_files(TEMP_DIR)
        cleanup_old_files(PROCESSED_DIR)
        
        # 保存上传的文件
        file_path = save_upload_file(file)
        
        try:
            # 提取PDF文本
            resume_text = extract_text_from_pdf(str(file_path))
            
            # 计算匹配分数
            match_score = calculate_match_score(resume_text, job_role)
            
            # 获取改进建议
            suggestions = improve_resume(resume_text, job_role)
            
            # 处理完成后移动到processed目录
            processed_path = PROCESSED_DIR / file_path.name
            shutil.move(str(file_path), str(processed_path))
            
            return {
                "match_score": f"{match_score:.2f}%",
                "suggestions": suggestions
            }
            
        except Exception as e:
            # 发生错误时删除临时文件
            if file_path.exists():
                file_path.unlink()
            raise e
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理简历时出错：{str(e)}")
