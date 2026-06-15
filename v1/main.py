# main.py
import os
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

app = FastAPI(title="Universal Web Extractor AI")
templates = Jinja2Templates(directory="templates")

# =====================================================================
# GLOBAL INSTANTIATION
# Declared directly in the main body (not under def) for connection pooling
# =====================================================================
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://placeholder-endpoint.openai.azure.com/"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY", "placeholder-key"),
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini"), 
    api_version="2024-08-01-preview",
    temperature=0.0
)

# Enforced Pydantic response format constraint schema
class ExtractionOutput(BaseModel):
    page_title: str = Field(description="The primary name or title of the scanned webpage.")
    scraped_summary: str = Field(description="Detailed overview answering the user prompt precisely.")
    key_insights: list[str] = Field(description="Bullet points containing structural insights extracted from the page.")
    data_quality_score: float = Field(description="Confidence rating from 0.0 to 1.0 based on page contents found.")

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None, "error": None})

@app.post("/extract", response_class=HTMLResponse)
async def handle_extraction(request: Request, url: str = Form(...), user_prompt: str = Form(...)):
    # Fallback runtime safety check to make sure keys aren't placeholders
    if "placeholder" in str(llm.azure_endpoint) or not os.getenv("AZURE_OPENAI_API_KEY"):
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": None, 
            "error": "Backend Error: Please check that your active terminal session has your real keys exported."
        })

    try:
        # Spawn the isolated browser sub-process inside the route request
        mcp_client = MultiServerMCPClient({
            "browser": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@playwright/mcp@latest", "--isolated"]
            }
        })

        # Fetch the tools dynamically from the fresh connection
        browser_tools = await mcp_client.get_tools()
        
        # Compile the modern agent factory harness referencing the global 'llm'
        agent = create_agent(
            model=llm, 
            tools=browser_tools, 
            response_format=ExtractionOutput
        )

        full_query = f"Go to {url} immediately. Once page finishes loading, follow this prompt request: {user_prompt}"

        # Invoke the agent asynchronously
        agent_response = await agent.ainvoke({
            "messages": [{"role": "user", "content": full_query}]
        })

        # Capture the typed structural object output
        structured_data: ExtractionOutput = agent_response.get("structured_response")

        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": structured_data,
            "error": None,
            "submitted_url": url,
            "submitted_prompt": user_prompt
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "result": None, "error": str(e)})
