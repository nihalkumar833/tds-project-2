# 📊 Multi-Modal Data Analysis API v1.0

<!-- Core Stack -->
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI%20Server-499848)](https://www.uvicorn.org/)

<!-- Playwright -->
[![Playwright](https://img.shields.io/badge/Playwright-1.54-blue?logo=microsoftedge&logoColor=white)](https://playwright.dev/docs/docker)

<!-- Data & Tools -->
[![Pandas](https://img.shields.io/badge/Pandas-2.3%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.2%2B-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-HTTPFS%20%26%20Parquet-ffc107?logo=duckdb&logoColor=000)](https://duckdb.org/)
[![tabula-py](https://img.shields.io/badge/tabula--py-Java%208%2B-ff6f00)](https://tabula-py.readthedocs.io/en/latest/getting_started.html)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

<!-- License & Repo -->
[![MIT License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github&logoColor)


***

## ✨ Features

- 📂 Multiple file uploads: CSV, JSON, HTML, PDF, images, archives (ZIP/TAR) parsed automatically.[7][8]
- 📝 Accepts free-text questions or questions.txt to drive the workflow.[8][7]
- 🌐 Web scraping via Playwright with stealth and HTML table extraction.[8][5]
- 📄 PDF table extraction and grouping via tabula-py (Java required).[7][6]
- 🧮 Numeric column auto-cleaning (currencies, percentages, scientific) with LLM guidance and heuristics.[8]
- 🗃 DuckDB schema preview and tiny samples for CSV/JSON/Parquet/HTTP/S3 sources.[7]
- 🔁 Multi-provider LLM fallback: Gemini→OpenAI for task breaking and code generation.[7][8]
- 🩺 Built-in health and provider test endpoints (/health, /test_gemini, /test_openai_call).[7]
- 🧱 Archive-safe extraction with path traversal guards and categorization.[7]

***

## 🗂 Available Workflows

| Workflow ID | Purpose |
|-------------|---------|
| data_analysis | General analysis & recommendations from uploaded or scraped tabular data[7][8] |
| text_analysis | Extract OCR text and augment questions for analysis[7] |
| web_scraping | Extract main data table(s) from target pages and save CSVs[8] |
| pdf_table_extraction | Extract, group, and merge tables from PDFs[7] |
| database_analysis | Lightweight schema/sample introspection via DuckDB for CSV/JSON/Parquet/HTTP/S3[7] |
| code_generation | LLM-generated Python analysis script with constrained execution[7] |

Note: Heavy tasks may be slow locally; for production hardening see Deployment notes below.[2][3][6][5]

***

## 🔌 API Endpoints

| Method | Endpoint     | Description |
|--------|--------------|-------------|
| POST   | /api/        | Submit analysis tasks with file uploads and/or text |
| GET    | /health      | System/process metrics and key config diagnostics |
| GET    | /            | Route listing and service info |
| GET    | /test_gemini_pro | Test Gemini multi-key/multi-model path |
| GET    | /test_gemini | Test Gemini multi-key/Pro-model path |
| GET    | /test_openai_call | Test OpenAI client fallback path |
| GET    | /test_ocr_call | Minimal live OCR call diagnostics |
| GET    | /system_diagnostics | Ultra-Advanced Project Diagnostics |

***

## 📥 Example Request (cURL)

```bash
curl "http://localhost:8000/api/" \
  -F "questions=@questions.txt" \
  -F "files=@data.csv" \
  -F "files=@report.pdf" \
  -F "files=@archive.zip"
```

- Optional: send no files and just a question text body; the service can extract URLs and attempt scraping where appropriate.[8][7]
- For images/PDFs, OCR and tabula are used if configured and available.

***

## 🖥 Local Setup

### 1️⃣ Install system dependencies
- Java 8+ required for tabula-py.
- Optional Playwright browsers: see Docker or run playwright install inside venv.

```bash
# Ubuntu/Debian (Java + build essentials)
sudo apt-get update
sudo apt-get install -y default-jre build-essential python3-dev libssl-dev libxml2-dev libxslt-dev zlib1g-dev
```

### 2️⃣ Create environment and install Python deps
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3️⃣ Configure environment
```bash
cp .env.template .env
# Edit .env and add your API keys (e.g., OPENAI_API_KEY, gemini_api_1, OCR_API_KEY)
```

### 4️⃣ Run locally (development)
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Production guidance recommends running behind a process manager (e.g., Gunicorn with multiple Uvicorn workers).[3][2][4]

***

## 🐳 Docker (with Playwright/Chromium)

Running Playwright in containers benefits from specific flags: use --init to avoid zombies and --ipc=host for Chromium stability; prefer non-root user to preserve sandboxing.[5]

```bash
# Example Dockerfile base (outline)
# FROM mcr.microsoft.com/playwright:v1.54.0-noble
# RUN apt-get update && apt-get install -y default-jre && rm -rf /var/lib/apt/lists/*
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . .
# EXPOSE 8000
# CMD ["gunicorn","-k","uvicorn.workers.UvicornWorker","-w","4","-t","120","app:app","--bind","0.0.0.0:8000"]

# Build
docker build -t multi-modal-api .

# Run (Playwright recommendations)
docker run --rm -p 8000:8000 --init --ipc=host --env-file .env multi-modal-api
```

- If running browsers as root, Chromium sandbox is disabled; to keep sandboxing, run as non-root user in the image.[5]
- Ensure Java is present for tabula-py in the image.[6]

***

## 🌍 VM Deployment (Linux)

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y git docker.io python3 python3-pip default-jre
sudo systemctl enable docker && sudo systemctl start docker
git clone https://github.com/USERNAME/REPO.git
cd REPO
cp .env.template .env && nano .env   # Add API keys
docker build -t multi-modal-api .
docker run --rm -d -p 8000:8000 --init --ipc=host --env-file .env multi-modal-api
```

For direct ASGI serving without Docker, see FastAPI deployment concepts and manual server docs; typically use Gunicorn with Uvicorn workers in production.

***

## 🔍 Test in Browser

- Health check: http://localhost:8000/health
- Routes: http://localhost:8000/
- Swagger UI (if enabled by your setup): http://localhost:8000/docs

***

## ⚙ Environment Variables

| Name                     | Required | Description |
|--------------------------|----------|-------------|
| OPENAI_API_KEY           | Required | Used for OpenAI fallback in code generation and Q&A  |
| gemini_api_1..gemini_api_10 | Required | Gemini multi-key rotation for task breaking and extraction |
| OCR_API_KEY              | Required | OCR.space key for image/PDF text extraction |
| HORIZON_API              | Optional | Custom integration key (if used by your flows) |
| GROK_API, GROK_FIX_API   | Optional | Custom integration keys (if used by your flows) |

Note: If Java is missing, PDF extraction will be disabled/fail at runtime; ensure Java 8+ is installed for tabula-py.[6]

***

## 📄 Request Guidelines

- Provide a questions.txt or include questions in a text form field to steer analysis.[7]
- Upload relevant data files (CSV/JSON/HTML/PDF) and/or compressed archives; the service categorizes and processes contents automatically.[7]
- For web scraping, include explicit URLs in the question text; the service attempts to extract data tables from those pages.[8][7]
- For databases/Parquet/CSV over HTTP/S3, the service previews schemas using DuckDB with httpfs/parquet extensions loaded at runtime.[7]

***

## 🛡 Production Notes

- Use a production process model (Gunicorn+Uvicorn workers) and set reasonable timeouts.[3][4][2]
- Running Playwright in Docker: add --init and --ipc=host; run as non-root to keep Chromium sandboxing.[5]
- Ensure Java 8+ is installed for tabula; consider setting tabula java_options (e.g., -Xmx512m) and page limits for large PDFs.[6]
- Consider background workers for long-running scraping/OCR/PDF tasks to keep the API responsive.[2][3]

***

## 📜 License

This project is licensed under the MIT License. See LICENSE.


***



