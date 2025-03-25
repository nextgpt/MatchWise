# 🚀 MatchWise – AI-Powered Resume Matching  

🔍 **MatchWise** helps job seekers optimize their resumes by analyzing job descriptions and providing a **match score** along with AI-powered **improvement suggestions**.  

## 🛠️ Tools & Technologies Used  

- 🤖 **PEFT (Parameter-Efficient Fine-Tuning)** – Fine-tuned **BERT** for similarity scoring  
- 📝 **Google Gemini AI** – Generates resume improvement suggestions  
- 🏗️ **FastAPI** – Lightweight and fast backend API  
- 📄 **PyMuPDF (fitz)** – Extracts text from resume PDFs  
- 🔢 **PyTorch** – Computes job-resume similarity  
- 📦 **Transformers (Hugging Face)** – Loads BERT-based model  
- 🐍 **Python 3.11+** – Programming language  
- 📂 **Docker (Optional)** – For containerized deployment  

---

## 🏗️ Project Workflow  

1️⃣ **Upload Resume (PDF)** 📝  
2️⃣ **Extract Text from PDF** 📄  
3️⃣ **Compare Resume & Job Role** 🏗️  
4️⃣ **Compute Match Score (BERT + PEFT)** 📊  
5️⃣ **Get AI-Powered Resume Suggestions** 🧠  
6️⃣ **Return Results via API** 🔄  

---

## 🚀 Installation & Setup  

### 🔹 Prerequisites  

Ensure you have the following installed:  

- Python 3.11+  
- pip package manager  
- Virtual environment (optional)  

### 🔹 Clone Repository  

```sh
git clone https://github.com/namanomar/MatchWise.git
cd MatchWise
```

### 🔹 Install Dependencies

```sh
pip install -r requirements.txt
```

🔹 Set Up Environment Variables

Create a .env file and add your Gemini API key:

```sh
GEMINI_API_KEY=your_api_key_here
```


🔹 Run the FastAPI Server

```sh
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
🔗 http://127.0.0.1:8000/docs

🔥 API Endpoints
1️⃣ Upload Resume & Get Match Score
Endpoint: POST /match-resume/

Params:

file: Resume PDF 📄

job_role: Desired job role 💼

Response:
```
{
  "match_score": "85.67%",
  "suggestions": "1. Add more details on your Python experience. 2. Mention machine learning projects."
}
```

🎯 Future Enhancements
✅ Support for multiple job descriptions

✅ Web UI for user-friendly interaction

✅ Advanced AI Resume Writing