# CodeStructure.md

This document explains the main classes, methods, and responsibilities across the key files:

- `main.py`
- `chains/workflows.py`
- `chains/base.py`
- `chains/web_scraping_steps.py`

---

## `main.py`

**Purpose**

Entry point for the FastAPI application. Handles request validation, workflow detection, orchestrator initialization, and the main `/api/` endpoint.

**Key Pydantic models**

- `TaskRequest` — request model for analysis tasks.
- `WorkflowRequest` — request model for single workflow executions.
- `MultiStepWorkflowRequest` — model for multi-step pipelines.
- `TaskResponse` — standardized response model for tasks.

**Important functions**

- `extract_output_requirements(task_description)`  
  Parse the task text to detect required formats (JSON/CSV/table), whether visualizations are requested, size limits, and base64 requirements.

- `detect_workflow_type_llm(task_description, default_workflow)`  
  Primary workflow classifier using an LLM prompt. Falls back to keyword detection if LLM unavailable.

- `detect_workflow_type_fallback(task_description, default_workflow)`  
  Keyword-based fallback for mapping tasks to workflows (e.g., web_scraping, image_analysis, data_visualization).

- `prepare_workflow_parameters(task_description, workflow_type, file_content)`  
  Build the `parameters` dict used by workflows: URL extraction, data-type hints, chart preferences, output format, file content type detection, etc.

- FastAPI endpoints:
  - `GET /` — API info
  - `GET /health` — health + orchestrator status
  - `POST /api/` — file + questions handler that runs the detected workflow synchronously (≤ 3 minutes)

**Orchestrator init**

Tries to initialize `AdvancedWorkflowOrchestrator`. If it fails, falls back to a `MinimalOrchestrator` that registers a basic `ModularWebScrapingWorkflow` so the API can still respond.

---

## `chains/workflows.py`

**Purpose**

Collection of domain-specific workflows and the central orchestrator that registers and runs them.

**Key classes (each implements `async execute()`):**

- `ExploratoryDataAnalysisWorkflow`
- `DataAnalysisWorkflow`
- `ImageAnalysisWorkflow`
- `CodeGenerationWorkflow`
- `PredictiveModelingWorkflow`
- `DataVisualizationWorkflow`
- `WebScrapingWorkflow`
- `DatabaseAnalysisWorkflow`
- `StatisticalAnalysisWorkflow`
- `MultiStepWebScrapingWorkflow`
- `ModularWebScrapingWorkflow`

Each workflow inherits from `BaseWorkflow` and provides domain-specific execution logic, prompting, and tool usage.

**Orchestration**

- `AdvancedWorkflowOrchestrator` — registers workflows, exposes `execute_workflow()` to run a workflow by name, manages execution history and LLM components.
- `STEP_REGISTRY` — mapping of step names → web scraping step classes (used by modular scrapers).
- Utility orchestrator helpers:
  - `run_web_scraping_workflow()`
  - `detect_steps_from_prompt()`
  - `run_llm_planned_workflow()`

These helpers enable modular, LLM-driven multi-step execution (detect plan → execute steps → collect results).

---

## `chains/base.py`

**Purpose**

Foundational abstractions used by workflows and chains (LLM initialization, shared utilities, common prompts).

**Key classes**

- `BaseWorkflow`  
  Abstract base for workflows. Manages:
  - model/LLM initialization
  - standard prompt templates
  - optional memory or tool injection  
  Requires derived classes to implement `async execute(self, input_data)`.

- `WorkflowOrchestrator`  
  Base orchestrator that:
  - registers workflows
  - routes requests
  - maintains execution logs and simple history

**Specialized chains**

- `DataAnalysisChain` — builder for data-analysis prompt → chain execution logic
- `CodeGenerationChain` — creates code-generation prompts and executes them safely
- `ReportGenerationChain` — compiles results into a human-ready report

These encapsulate prompt templates, output parsers, and chain wiring for reuse across workflows.

---

## `chains/web_scraping_steps.py`

**Purpose**

A modular, step-based pipeline for generic web scraping, table extraction, cleaning, analysis, visualization, and question answering. Each step is designed to be LLM-assisted with robust fallbacks.

**Step classes & roles**

1. `DetectDataFormatStep`  
   - Decide extraction strategy for a given URL / HTML (table, JSON, JS-rendered content, page fragments).
   - Uses LLM + heuristic fallbacks to return a recommended extraction plan.

2. `ScrapeTableStep`  
   - Execute extraction based on the plan (e.g., `pandas.read_html`, JSON parsing, or custom DOM traversal).
   - Returns raw table(s) and basic metadata.

3. `InspectTableStep`  
   - Inspect extracted tables for header rows, MultiIndex columns, and whether first row is header.
   - Normalizes column names and restructures table if necessary.

4. `CleanDataStep`  
   - Clean values (remove currency symbols, footnotes, scale suffixes like M/B, non-numeric chars).
   - Convert numeric columns, handle missing values, and detect summary/total rows.

5. `AnalyzeDataStep`  
   - Choose relevant numeric columns for the task (LLM-assisted).
   - Compute statistics, correlations, top-N lists, aggregates, etc.
   - Filters out summary rows and returns analyzed dataframe + insights.

6. `VisualizeStep`  
   - Auto-detect appropriate chart (scatter, bar, time-series, histogram).
   - Plot using Matplotlib and encode images as base64 data URIs (respecting max size constraints).
   - Optionally include regression lines or other overlays as requested.

7. `AnswerQuestionsStep`  
   - Use the cleaned and analyzed data to answer the user's questions with LLM assistance.
   - Produce final JSON-formatted answers, embed base64 charts if required.

**Design**

- Each step provides `run(input_data)` and returns structured output consumed by the next step.
- Steps include small helper methods for LLM prompts, heuristics, and safe fallbacks to pure-Python logic when LLM is unavailable.
- The pipeline is **generic** and **domain-adaptive** — table selection, cleaning rules, and chart choices are LLM-guided rather than hard-coded.

---

## Notes & Best Practices

- **LLM-driven decisions:** Key choices (table selection, header detection, cleaning strategy, chart type) are done via LLM prompts with deterministic fallbacks.
- **No hardcoded column names:** Workflows must detect and reference columns dynamically (e.g., `df.columns[0]`) rather than assuming names.
- **Fallback-first mindset:** Orchestrator and steps always provide non-LLM fallbacks so the system remains usable offline or when a model fails.
- **Image size limit:** Visual outputs encoded as base64 should respect configured max size (e.g., ≤ 100KB).
- **Synchronous timeout:** API-level timeout is enforced (default ≤ 3 minutes) to return a timely response.

---

## Quick reference

- **Entry point:** `main.py`
- **Orchestrator & registration:** `chains/workflows.py`
- **Base classes / shared chains:** `chains/base.py`
- **Web scraping & single-page ETL pipeline:** `chains/web_scraping_steps.py`

---

Which would you like next?
