# 📊 Multi-Modal Data Analysis API v2.0

<!-- Core Stack -->
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI%20Server-499848)](https://www.uvicorn.org/)

<!-- AI & LLM -->
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-blue?logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)](https://faiss.ai/)

<!-- Data & Tools -->
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.25+-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

<!-- License & Repo -->
[![MIT License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/)


> 🚀 AI-powered API for text, image, and code analysis — upload your files & get **instant insights**.

---

## ✨ Features
- 📂 **Multiple file uploads** (CSV, images, JSON…)
- 📝 **Required:** `questions.txt` with your analysis questions
- ⚡ **Synchronous processing** — results in ≤ 3 min
- 🤖 **AI workflow detection** (no manual config)
- 📊 **10+ ready-to-use workflows**
- 🛠 **FastAPI + LangChain + OpenAI/Gemini support**

---

## 🗂 Available Workflows
| Workflow ID | Purpose |
|-------------|---------|
| `data_analysis` | General analysis & recommendations |
| `image_analysis` | Computer vision tasks |
| `text_analysis` | Natural language processing |
| `code_generation` | Python code creation |
| `exploratory_data_analysis` | EDA planning & execution |
| `predictive_modeling` | Machine learning guidance |
| `data_visualization` | Charts & graphs |
| `statistical_analysis` | Stats & correlations |
| `web_scraping` | Web data extraction |
| `database_analysis` | SQL & DuckDB queries |

---

## 🔌 API Endpoints
| Method | Endpoint   | Description |
|--------|-----------|-------------|
| POST   | `/api/`   | Submit analysis tasks (upload files) |
| GET    | `/health` | Health check |
| GET    | `/`       | API info |

---

## 📥 Example Request (cURL)
```bash
curl "http://localhost:8000/api/" \
  -F "questions_txt=@questions.txt" \
  -F "files=@data.csv" \
  -F "files=@image.png"
````

---

## 🖥 Local Setup

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Configure environment

```bash
cp .env.template .env
# Open .env and add your API keys
```

### 3️⃣ Run locally

```bash
uvicorn main:app --reload
```

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t data-analysis-api .

# Run the container
docker run -d -p 8000:80 --env-file .env data-analysis-api
```

---

## 🌍 VM Deployment (Linux)

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y git docker.io vim python3 python3-pip
sudo systemctl enable docker && sudo systemctl start docker
git clone https://github.com/USERNAME/REPO.git
cd REPO/Project2
cp .env.template .env
vim .env   # Add your API keys
bash run_docker.sh
```

---

## 🔍 Test in Browser

* Upload files: **`https://tds-project-2-production.up.railway.app/static/test_upload.html`**
* Health check: **`https://tds-project-2-production.up.railway.app/health`**

---

## ⚙ Environment Variables

| Name                   | Required | Description              |
| ---------------------- | -------- | ------------------------ |
| `OPENAI_API_KEY`       | ✅ Yes    | Your OpenAI API key      |
| `GEMINI_API_KEY`       | ✅ Yes    | Your OpenAI API key      |
| `LANGCHAIN_API_KEY`    | ❌ No     | For LangSmith tracing    |
| `LANGCHAIN_TRACING_V2` | ❌ No     | Enable LangSmith tracing |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---
