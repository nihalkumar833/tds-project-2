# 📊 Data Analyst Agent

> A powerful API that uses LLMs to **source, prepare, analyze, and visualize** data — all from a simple POST request.

🌐 **Live App:** [https://tds-project-2-production.up.railway.app](https://tds-project-2-production.up.railway.app)  
📄 **Project Brief:** [View Details](https://tds.s-anand.net/#/project-data-analyst-agent?id=project-data-analyst-agent)

---

## 🚀 Overview

This project deploys a **Data Analyst Agent API**.  
You send it a question → it fetches and analyzes data → it sends back the answer with optional visualizations.

Example request:

```bash
curl "https://surprising-mindfulness.onrender.com/api/" \
     -F "@question.txt"
````

---

## 🛠 Features

* **Natural Language Analysis** – Understands complex questions.
* **Data Sourcing** – Fetches data from the web, APIs, or cloud storage.
* **Data Preparation** – Cleans and transforms datasets.
* **Statistical Analysis** – Runs correlations, regressions, and more.
* **Visualization** – Generates plots and returns them as base64 image URIs.
* **Fast Responses** – Designed to answer within **3 minutes**.

---

## 📌 Example Use Cases

### 1️⃣ Wikipedia Data

**Question:**
*How many \$2bn movies were released before 2020?*

**Data Source:**
[https://en.wikipedia.org/wiki/List\_of\_highest-grossing\_films](https://en.wikipedia.org/wiki/List_of_highest-grossing_films)

---

### 2️⃣ Court Judgement Analysis

**Question:**
*Which high court disposed of the most cases from 2019–2022?*

**Data Source:**
Indian High Court Judgement Dataset ([ecourts.gov.in](https://judgments.ecourts.gov.in/))

---

## 🧰 Tech Stack

* **Backend:** FastAPI + Uvicorn
* **LLM Integration:** OpenAI API, LangChain
* **Data Processing:** Pandas, DuckDB, PyArrow
* **Visualization:** Matplotlib
* **Deployment:** Docker, Render

---

## 📦 Installation

```bash
git clone https://github.com/your-username/data-analyst-agent.git
cd data-analyst-agent
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## 🌍 Deployment

This project is **Dockerized** and ready for cloud deployment.
You can host it on **Render**, **Heroku**, or **AWS Lambda**.

---

## 🏷 Suggested Topics

```
fastapi langchain docker web-scraping data-analysis llm ai api visualization pandas matplotlib openai
```

---

## 📜 License

MIT License – free to use and modify.

---

