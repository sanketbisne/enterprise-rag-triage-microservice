from langchain_core.tools import tool
import json
import os
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud import storage

# Integrates with a real Vertex AI Search Data Store
@tool
def search_knowledge_base(query: str) -> str:
    """Useful for searching the company knowledge base for policy, process, and general information."""
    print(f"[Tool] Searching Vertex AI Search for: {query}")
    
    # Query Expansion for common acronyms
    original_query = query
    if "GDE" in query.upper():
        query += " Google Developer Expert"
        print(f"[Tool] Expanded query: {query}")

    project_id = os.getenv("GCP_PROJECT_ID", "mcp-gcp-project")
    location = os.getenv("VERTEX_SEARCH_LOCATION", "global")
    data_store_id = os.getenv("VERTEX_SEARCH_DATA_STORE_ID")
    
    if not data_store_id:
        return "Error: VERTEX_SEARCH_DATA_STORE_ID environment variable not set."

    try:
        client = discoveryengine.SearchServiceClient()
        doc_client = discoveryengine.DocumentServiceClient()
        storage_client = storage.Client()
        serving_config = client.serving_config_path(
            project=project_id,
            location=location,
            data_store=data_store_id,
            serving_config="default_config",
        )
        
        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query,
            page_size=5, # Increased page size for better recall
        )
        
        response = client.search(request)
        print(f"[Debug Tool] Vertex Search returned {len(list(response.results))} results.")
        
        # Reset results iterator for the loop
        response = client.search(request)
        
        answers = []
        
        def find_text_fields(data, found_texts):
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, str) and len(v) > 20: # Heuristic for content
                        found_texts.append(v)
                    else:
                        find_text_fields(v, found_texts)
            elif isinstance(data, list):
                for item in data:
                    find_text_fields(item, found_texts)

        for result in response.results:
            document = result.document
            document_data = document.derived_struct_data
            
            if "extractive_answers" in document_data:
                answers.append(document_data["extractive_answers"][0].get("content", ""))
            elif "snippets" in document_data:
                answers.append(document_data["snippets"][0].get("snippet", ""))
            else:
                # Try to fetch from GCS link if provided
                content_found = False
                if "link" in document_data and document_data["link"].startswith("gs://"):
                    try:
                        gcs_path = document_data["link"]
                        print(f"[Debug Tool] Fetching from GCS: {gcs_path}")
                        bucket_name = gcs_path.split("/")[2]
                        blob_name = "/".join(gcs_path.split("/")[3:])
                        bucket = storage_client.bucket(bucket_name)
                        blob = bucket.blob(blob_name)
                        answers.append(blob.download_as_text())
                        content_found = True
                    except Exception as e:
                        print(f"[Debug Tool] GCS Fetch failed: {str(e)}")
                
                if not content_found:
                    # Try GetDocument as secondary fallback
                    try:
                        full_doc = doc_client.get_document(name=document.name)
                        if full_doc.content and full_doc.content.raw_bytes:
                            answers.append(full_doc.content.raw_bytes.decode("utf-8", errors="ignore"))
                            content_found = True
                        elif full_doc.struct_data:
                            found_texts = []
                            find_text_fields(dict(full_doc.struct_data), found_texts)
                            if found_texts:
                                answers.extend(found_texts)
                                content_found = True
                    except Exception:
                        pass
        
        final_context = "\n".join(filter(None, answers))
        
        if not final_context or final_context.strip() == "":
             # FINAL FALLBACK: Check local file if search returned nothing
             if os.path.exists("knowledge_base.txt"):
                 print("[Debug Tool] Falling back to local knowledge_base.txt")
                 with open("knowledge_base.txt", "r") as f:
                     return f.read()
             return "No matching information found in the Vertex AI Search knowledge base."
            
        return final_context
        
    except Exception as e:
        # Error-time fallback
        if os.path.exists("knowledge_base.txt"):
             with open("knowledge_base.txt", "r") as f:
                 return f.read()
        return f"Vertex AI Search failed: {str(e)}"

# This mimics integrating with an enterprise CRM database
@tool
def lookup_customer(customer_id: str) -> str:
    """Look up a customer's profile and active services based on their ID."""
    print(f"[Tool] Looking up CRM for: {customer_id}")
    
    # Mock CRM data
    crm = {
        "CUST-101": {"name": "Alice Smith", "tier": "Enterprise", "active_services": ["Cloud Run", "Vertex AI"]},
        "CUST-202": {"name": "Bob Jones", "tier": "Standard", "active_services": ["Cloud Storage"]}
    }
    
    customer = crm.get(customer_id)
    if customer:
        return json.dumps(customer)
    return f"Customer ID {customer_id} not found."

# This mimics an action that mutates state (e.g. hitting Jira/ServiceNow API)
@tool
def create_escalation_ticket(issue_description: str, customer_id: str = "Unknown") -> str:
    """Create a Tier 2 support escalation ticket. Use only when the issue cannot be resolved via knowledge base."""
    print(f"[Tool] CREATING TICKET for {customer_id}: {issue_description}")
    
    ticket_id = f"TICK-{hash(issue_description) % 10000}"
    return f"Success! Escalation ticket {ticket_id} has been created for {customer_id}."

# Combine all tools for the agent
tools = [search_knowledge_base, lookup_customer, create_escalation_ticket]
