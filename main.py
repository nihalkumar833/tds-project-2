from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
import asyncio
import logging
import os
import sys
import aiofiles

# ==== Import your project modules ====
from utils.prompts import WORKFLOW_DETECTION_SYSTEM_PROMPT, WORKFLOW_DETECTION_HUMAN_PROMPT
from utils.constants import (
    VALID_WORKFLOWS, API_TITLE, API_DESCRIPTION, API_VERSION, API_FEATURES,
    API_ENDPOINTS, STATUS_OPERATIONAL, STATUS_HEALTHY, STATUS_AVAILABLE,
    STATUS_UNAVAILABLE, LOG_FORMAT, LOG_FILE, STATIC_DIRECTORY, STATIC_NAME,
    DEFAULT_WORKFLOW, DEFAULT_PRIORITY, DEFAULT_TARGET_AUDIENCE, DEFAULT_PIPELINE_TYPE,
    DEFAULT_OUTPUT_REQUIREMENTS, SCRAPING_KEYWORDS, MULTI_STEP_KEYWORDS,
    IMAGE_KEYWORDS, TEXT_KEYWORDS, LEGAL_KEYWORDS, STATS_KEYWORDS, DB_KEYWORDS,
    VIZ_KEYWORDS, EDA_KEYWORDS, ML_KEYWORDS, CODE_KEYWORDS, WEB_KEYWORDS,
    DATA_TYPE_FINANCIAL, DATA_TYPE_RANKING, DATABASE_TYPE_SQL, FILE_FORMAT_PARQUET,
    CHART_TYPE_SCATTER, OUTPUT_FORMAT_BASE64, MAX_FILE_SIZE,
    CONTENT_TYPE_JSON, CONTENT_TYPE_CSV, CONTENT_TYPE_TEXT,
    PLOT_CHART_KEYWORDS, FORMAT_KEYWORDS, KEY_INCLUDE_VISUALIZATIONS, KEY_VISUALIZATION_FORMAT,
    KEY_MAX_SIZE, KEY_FORMAT, VISUALIZATION_FORMAT_BASE64, MAX_SIZE_BYTES,
    FINANCIAL_DETECTION_KEYWORDS, RANKING_DETECTION_KEYWORDS, DATABASE_DETECTION_KEYWORDS,
    CHART_TYPE_KEYWORDS, REGRESSION_KEYWORDS, BASE64_KEYWORDS, URL_PATTERN, S3_PATH_PATTERN
)

# ==== Setup paths ====
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chains"))

# ==== Configure logging ====
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)]
)
logger = logging.getLogger(__name__)

# ==== Initialize orchestrator ====
try:
    from chains.workflows import AdvancedWorkflowOrchestrator
    orchestrator = AdvancedWorkflowOrchestrator()
    logger.info("AdvancedWorkflowOrchestrator initialized successfully.")
except Exception as e:
    logger.error(f"Could not import or initialize workflows: {e}")
    try:
        from chains.workflows import ModularWebScrapingWorkflow
        from chains.base import WorkflowOrchestrator
        class MinimalOrchestrator(WorkflowOrchestrator):
            def __init__(self):
                super().__init__()
                self.llm = None
                self.workflows = {"multi_step_web_scraping": ModularWebScrapingWorkflow()}
        orchestrator = MinimalOrchestrator()
        logger.info("Created minimal orchestrator with fallback workflows")
    except Exception as e2:
        logger.error(f"Could not create minimal orchestrator: {e2}")
        orchestrator = None

# ==== FastAPI App ====
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)
if os.path.isdir(STATIC_DIRECTORY):
    app.mount(f"/{STATIC_NAME}", StaticFiles(directory=STATIC_DIRECTORY), name=STATIC_NAME)

# ==== Health and Root Routes ====
@app.get("/")
async def root():
    return {
        "message": f"{API_TITLE} v{API_VERSION}",
        "description": API_DESCRIPTION,
        "features": API_FEATURES,
        "endpoints": API_ENDPOINTS,
        "status": STATUS_OPERATIONAL,
    }

@app.get("/health")
async def health_check():
    return {
        "status": STATUS_HEALTHY,
        "timestamp": datetime.now().isoformat(),
        "orchestrator": STATUS_AVAILABLE if orchestrator else STATUS_UNAVAILABLE,
        "workflows_available": (len(orchestrator.workflows) if orchestrator else 0),
        "version": API_VERSION,
    }

# ==== Helper Functions ====
def extract_output_requirements(task_description: str) -> Dict[str, Any]:
    reqs = DEFAULT_OUTPUT_REQUIREMENTS.copy()
    task_lower = task_description.lower()
    if any(k in task_lower for k in PLOT_CHART_KEYWORDS):
        reqs[KEY_INCLUDE_VISUALIZATIONS] = True
        reqs[KEY_VISUALIZATION_FORMAT] = VISUALIZATION_FORMAT_BASE64
        reqs[KEY_MAX_SIZE] = MAX_SIZE_BYTES
    if any(k in task_lower for k in FORMAT_KEYWORDS):
        if "json" in task_lower:
            reqs[KEY_FORMAT] = CONTENT_TYPE_JSON
        elif "csv" in task_lower:
            reqs[KEY_FORMAT] = CONTENT_TYPE_CSV
        elif "table" in task_lower:
            reqs[KEY_FORMAT] = "table"
    return reqs

async def detect_workflow_type_llm(task_description: str, default_workflow: str = DEFAULT_WORKFLOW) -> str:
    if not task_description:
        return default_workflow
    try:
        if orchestrator and getattr(orchestrator, 'llm', None):
            from langchain_core.prompts import ChatPromptTemplate
            from langchain.chains import LLMChain
            prompt = ChatPromptTemplate.from_messages([
                ("system", WORKFLOW_DETECTION_SYSTEM_PROMPT),
                ("human", WORKFLOW_DETECTION_HUMAN_PROMPT),
            ])
            detected = LLMChain(llm=orchestrator.llm, prompt=prompt).run(task_description=task_description).strip().lower()
            if detected in VALID_WORKFLOWS:
                return detected
        return detect_workflow_type_fallback(task_description, default_workflow)
    except Exception as e:
        logger.error(f"Error in workflow detection: {e}")
        return detect_workflow_type_fallback(task_description, default_workflow)

def detect_workflow_type_fallback(task_description: str, default_workflow: str = DEFAULT_WORKFLOW) -> str:
    t = task_description.lower()
    if any(k in t for k in SCRAPING_KEYWORDS): return "multi_step_web_scraping"
    if any(k in t for k in IMAGE_KEYWORDS): return "image_analysis"
    if any(k in t for k in TEXT_KEYWORDS): return "text_analysis"
    if any(k in t for k in LEGAL_KEYWORDS): return "data_analysis"
    if any(k in t for k in STATS_KEYWORDS): return "statistical_analysis"
    if any(k in t for k in DB_KEYWORDS): return "database_analysis"
    if any(k in t for k in VIZ_KEYWORDS): return "data_visualization"
    if any(k in t for k in EDA_KEYWORDS): return "exploratory_data_analysis"
    if any(k in t for k in ML_KEYWORDS): return "predictive_modeling"
    if any(k in t for k in CODE_KEYWORDS): return "code_generation"
    if any(k in t for k in WEB_KEYWORDS): return "multi_step_web_scraping"
    return default_workflow

def prepare_workflow_parameters(task_description: str, workflow_type: str, file_content: str = None) -> Dict[str, Any]:
    params = {}
    t_lower = task_description.lower()
    if "http" in t_lower:
        import re
        params["target_urls"] = re.findall(URL_PATTERN, task_description)
    if any(k in t_lower for k in FINANCIAL_DETECTION_KEYWORDS):
        params["data_type"] = DATA_TYPE_FINANCIAL
    elif any(k in t_lower for k in RANKING_DETECTION_KEYWORDS):
        params["data_type"] = DATA_TYPE_RANKING
    if "s3://" in t_lower:
        import re
        params["s3_paths"] = re.findall(S3_PATH_PATTERN, task_description)
    if any(k in t_lower for k in DATABASE_DETECTION_KEYWORDS):
        params["database_type"] = DATABASE_TYPE_SQL
    if "parquet" in t_lower:
        params["file_format"] = FILE_FORMAT_PARQUET
    if any(k in t_lower for k in CHART_TYPE_KEYWORDS):
        params["chart_type"] = CHART_TYPE_SCATTER
    if any(k in t_lower for k in REGRESSION_KEYWORDS):
        params["include_regression"] = True
    if any(k in t_lower for k in BASE64_KEYWORDS):
        params["output_format"] = OUTPUT_FORMAT_BASE64
        params["max_size"] = MAX_FILE_SIZE
    if file_content:
        params["file_content_length"] = len(file_content)
        s = file_content.strip()
        if s.startswith(("{", "[")):
            params["content_type"] = CONTENT_TYPE_JSON
        elif "\t" in file_content or "," in file_content:
            params["content_type"] = CONTENT_TYPE_CSV
        else:
            params["content_type"] = CONTENT_TYPE_TEXT
    return params

# ==== Execute workflow ====
async def execute_workflow_sync(workflow_type: str, workflow_input: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    try:
        if orchestrator is None:
            return {
                "workflow_type": workflow_type,
                "status": "completed_fallback",
                "message": "Orchestrator not available",
                "parameters_prepared": workflow_input.get("parameters", {}),
                "files_processed": list(workflow_input.get("additional_files", {}).keys()),
            }
        if workflow_type not in orchestrator.workflows:
            return {
                "workflow_type": workflow_type,
                "status": "error",
                "message": f"Workflow {workflow_type} not found",
                "available_workflows": list(orchestrator.workflows.keys()),
            }
        return await orchestrator.execute_workflow(workflow_type, workflow_input)
    except Exception as e:
        logger.error(f"Error executing workflow {workflow_type}: {e}")
        raise

# ==== Main API Endpoint ====
@app.post("/api/")
async def analyze_data(request: Request):
    task_id = str(uuid.uuid4())
    try:
        # Parse form for flexible file picking
        form = await request.form()
        questions_upload = None
        additional_files = []
        for field_name, value in form.items():
            if hasattr(value, "filename") and value.filename:
                fname_lower = value.filename.lower()
                if "question" in fname_lower and fname_lower.endswith(".txt"):
                    questions_upload = questions_upload or value
                elif field_name.lower() in ("questions_txt", "question.txt"):
                    questions_upload = questions_upload or value
                else:
                    additional_files.append(value)
        if not questions_upload:
            raise HTTPException(status_code=400, detail="A questions file is required.")

        # Read question file
        q_bytes = await questions_upload.read()
        try:
            questions_text = q_bytes.decode("utf-8")
        except UnicodeDecodeError:
            questions_text = q_bytes.decode("latin-1", errors="replace")

        # Process other files
        processed_files = {}
        file_contents = {}
        for file in additional_files:
            b = await file.read()
            try:
                file_contents[file.filename] = b.decode("utf-8")
                is_text = True
            except UnicodeDecodeError:
                file_contents[file.filename] = f"Binary file: {file.filename} ({len(b)} bytes)"
                is_text = False
            processed_files[file.filename] = {
                "content_type": file.content_type,
                "size": len(b),
                "is_text": is_text
            }

        # Prepare workflow input
        detected_workflow = await detect_workflow_type_llm(questions_text, "multi_step_web_scraping")
        workflow_input = {
            "task_description": questions_text,
            "questions": questions_text,
            "additional_files": file_contents,
            "processed_files_info": processed_files,
            "workflow_type": detected_workflow,
            "parameters": prepare_workflow_parameters(questions_text, detected_workflow, questions_text),
            "output_requirements": extract_output_requirements(questions_text),
        }

        result = await asyncio.wait_for(
            execute_workflow_sync(detected_workflow, workflow_input, task_id),
            timeout=180
        )

        return {
            "task_id": task_id,
            "status": "completed",
            "workflow_type": detected_workflow,
            "result": result,
            "processing_info": {
                "questions_file": questions_upload.filename,
                "additional_files": list(processed_files.keys()),
                "workflow_auto_detected": True,
                "processing_time": "synchronous",
            },
            "timestamp": datetime.now().isoformat(),
        }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timed out after 3 minutes.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{task_id}] Fatal error")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
