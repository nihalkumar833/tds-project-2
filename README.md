
# 🌟 **TDS Project 2: Data Analyst Agent** — *AI-Powered Data Sidekick*

> **A smart, interactive, and beautiful way to analyze your data — powered by Google Generative AI & cutting-edge Python tools.**
> **Repo:** [📂 View on GitHub](https://github.com/23f1000805/tds-project-2)

---

## 📌 **Overview**

The **TDS Data Analyst Agent** transforms raw data into **actionable insights** in minutes.
Upload your dataset + questions, and get:

* 📊 **Interactive Visualizations**
* 🧠 **AI-Driven Insights**
* ⚡ **Automated Analysis Workflows**

Perfect for **business analysts, researchers, and data enthusiasts** who want **fast, accurate, and beautiful results** without manual crunching.

---

## ✨ **Features at a Glance**

| Feature                     | Description                                        |
| --------------------------- | -------------------------------------------------- |
| 🔍 **Intelligent Analysis** | Understands your data using Google's Generative AI |
| 📈 **Dynamic Charts**       | Visualizes data with Matplotlib & Seaborn          |
| 🌐 **Web Scraping**         | Pulls data from URLs in seconds                    |
| 📁 **Multi-Format Support** | Works with CSV, Excel, JSON, Parquet, TXT          |
| 🔄 **Batch Processing**     | Answers multiple questions in one go               |
| 🎨 **Modern UI**            | Clean, responsive, and beginner-friendly           |
| ⚡ **Real-Time Results**     | Progress tracking with fast computations           |

---

## 🚀 **Quick Start**

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/23f1000805/tds-project-2.git
cd tds-project-2
```

### **2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3️⃣ Set Environment Variables**

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_generative_ai_api_key
LLM_TIMEOUT_SECONDS=150
```

### **4️⃣ Run the App**

```bash
python app.py
```

Then open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 📖 **How to Use**

### **Step 1: Write Your Questions**

Create a `.txt` file:

```
What are the top-selling products?
Find correlation between variable X and Y
Show sales trends over the last 6 months
```

### **Step 2: Upload Your Data**

* **Required:** Questions file (`.txt`)
* **Optional:** Dataset in CSV/Excel/JSON/Parquet/TXT

### **Step 3: Get Your Insights**

* 🧮 **Processed by AI**
* 📊 **Visualized beautifully**
* 💡 **Actionable recommendations generated**

---

## 🛠 **Tech Stack**

**Backend**

* FastAPI 🚀 (Ultra-fast web framework)
* LangChain 🧠 (LLM orchestration)
* Google Generative AI ✨ (Smart insights)
* Pandas + NumPy 📊 (Data manipulation)
* Matplotlib + Seaborn 🎨 (Visualizations)

**Frontend**

* HTML5, CSS3, JavaScript
* Bootstrap-inspired styling for a professional look

---

## 🔧 **API Endpoints**

| Method | Endpoint   | Description                |
| ------ | ---------- | -------------------------- |
| `GET`  | `/`        | Main web interface         |
| `POST` | `/api` | Process questions + data   |
| `GET` | `/summary`  | Advanced Diagnosis of app |

---

## 📂 **Supported Data Formats**

| Format  | Extensions      |
| ------- | --------------- |
| CSV     | `.csv`          |
| Excel   | `.xlsx`, `.xls` |
| JSON    | `.json`         |
| Parquet | `.parquet`      |
| Text    | `.txt`          |

---

## 🎯 **Use Cases**

* **Business Intelligence** – Sales trends, customer insights
* **Research** – Statistical summaries, hypothesis testing
* **Data Science** – EDA, feature analysis, anomaly detection

---

## 🔒 **Security**

* Local data processing (no cloud storage)
* Environment variable protection for API keys
* Configurable CORS for production environments

---

## 🚀 **Deployment Options**

* **Local** → `python app.py`
* **Production** → `gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker`
* **Docker**

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "app.py"]
```

---

## 🤝 **Contributing**

We welcome PRs!

1. Fork the repo
2. Create a branch: `git checkout -b feature-name`
3. Commit + push
4. Submit PR 🚀

---

## 📜 **License**

Licensed under **MIT** — Free to use, modify, and share.

---

