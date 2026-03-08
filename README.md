# 🤖 Enterprise Agentic Triage System

[![Google Cloud Platform](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)](https://cloud.google.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-2D3748?style=for-the-badge&logo=langchain)](https://github.com/langchain-ai/langgraph)

An end-to-end, production-ready Agentic AI system built on **Google Cloud Platform**. This agent orchestrates complex customer support workflows, leveraging **Retrieval-Augmented Generation (RAG)** and specialized tools to triage enterprise inquiries with zero hallucinations.

---

## 🏗️ Architecture & Technology Stack

The agent follows a ReAct-style workflow:
1. **Analyze**: Gemma 2B evaluates the user query against system instructions.
2. **Search**: If a question involves people or policies, the agent invokes the `search_knowledge_base` tool.
3. **Retrieve**: Vertex AI Search fetches the most relevant context from GCS-hosted knowledge bases.
4. **Reason**: The agent synthesizes a response using **ONLY** the retrieved facts.

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) | State-machine based reasoning and tool-calling loops. |
| **LLM Engine** | **Gemma 2B** | Hosted in **Vertex AI Model Garden** with dedicated DNS routing. |
| **Knowledge Base** | **Vertex AI Search** | Enterprise-grade retrieval from company policy documents. |
| **API Framework** | [FastAPI](https://fastapi.tiangolo.com/) | High-performance async REST API. |
| **Deployment** | **Google Cloud Run** | Scalable, serverless hosting (4GiB RAM / 2 CPU). |

---

## 🚀 End-to-End Setup Guide: From Scratch

Follow these steps to set up the entire system in your Google Cloud environment.

### 1. Create Data Store & Knowledge Base
Provide the agent with its "memory" by connecting it to your company documents.

- **Initialize Data Store**: Go to **Vertex AI Search & Conversation**, click Create, and select **Cloud Storage**. Point it to your bucket containing `knowledge_base.txt`.
- **Configure Search App**: Link the Data Store to a new Search App. Enable "Extractive Content" in the Serving Config.

![GCP Data Store Setup](/Users/sanketbisne/.gemini/antigravity/brain/88e72f2b-591a-4854-8a12-cee33efbb98d/gcp_data_store_setup_1772996488070.png)

![Vertex AI Search Config](/Users/sanketbisne/.gemini/antigravity/brain/88e72f2b-591a-4854-8a12-cee33efbb98d/vertex_ai_search_config_1772996503209.png)

### 2. Deploy Gemma 2B (Model Garden)
The agent needs a reasoning engine. We use **Gemma 2B** for its speed and efficiency.

1. Navigate to **Vertex AI Model Garden**.
2. Search for **Gemma 2B** and click **Deploy**.
3. Capture the **Endpoint ID** and the **Dedicated DNS URL** (Private DNS) from the Deployment tab.

### 3. Deploy the Triage Agent (Cloud Run)
Deploy the FastAPI application that orchestrates the agent's logic.

1. **Environment Configuration**: Create a `.env` file:
   ```env
   GCP_PROJECT_ID=your-project-id
   GCP_LOCATION=us-central1
   VERTEX_SEARCH_DATA_STORE_ID=your-data-store-id
   VERTEX_MODEL_ENDPOINT_ID=your-endpoint-id
   VERTEX_DEDICATED_ENDPOINT=your-dedicated-dns-name
   ```

2. **Build and Push**:
   ```bash
   gcloud builds submit --tag europe-west4-docker.pkg.dev/[PROJECT_ID]/agent-repo/triage-agent .
   ```

3. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy triage-agent \
     --image europe-west4-docker.pkg.dev/[PROJECT_ID]/agent-repo/triage-agent \
     --region us-central1 \
     --memory 4Gi \
     --cpu 2 \
     --allow-unauthenticated \
     --set-env-vars [ENV_VARS]
   ```

![Cloud Run Deployment](/Users/sanketbisne/.gemini/antigravity/brain/88e72f2b-591a-4854-8a12-cee33efbb98d/cloud_run_deployment_success_1772996522055.png)

---

## 🎉 Final Result

Once deployed, you can interact with your agent. It will use the **Vertex AI Search** tool to provide grounded, hallucination-free answers about your policies and people.

![Agent Chat Preview](/Users/sanketbisne/.gemini/antigravity/brain/88e72f2b-591a-4854-8a12-cee33efbb98d/agent_chat_preview_1772996536241.png)

### Verification
```bash
curl -X POST [YOUR_CLOUD_RUN_URL]/triage \
-H "Content-Type: application/json" \
-d '{"message": "who is Sanket Bisne"}'
```

---

## 📊 API Endpoints

- `POST /triage`: Primary entry point for user messages.
- `GET /health`: Standard Liveness/Readiness probe.
- `GET /docs`: Interactive Swagger documentation.

---

## 👤 Author
**Sanket Bisne** - *Google Developer Expert (GDE)*
Specializing in Cloud, AI, and Enterprise Agentic Architectures.
