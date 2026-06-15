# main.py
import os
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# =====================================================================
# 1. SECURE ENVIRONMENT LOAD
# This reads the local .env file before the rest of the script executes
# =====================================================================
load_dotenv()

from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

app = FastAPI(title="Universal Web Extractor AI")

# =====================================================================
# 2. CORS SECURITY CONFIGURATION
# Protects your API from unauthorized cross-origin browser requests
# =====================================================================
origins = [
    "http://localhost:8500",      # Local dev connection
    "http://127.0.0.1:8500",     # Local loopback connection
    # "https://yourdomain.com",  # Add your production domain here when ready
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Only allow domains in our whitelist
    allow_credentials=True,
    allow_methods=["GET", "POST"],    # Strict HTTP method restriction
    allow_headers=["*"],              # Allows standard browser headers
)

# Template Directory Resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# Global LLM Instantiation via securely loaded variables
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini"), 
    api_version="2024-08-01-preview",
    temperature=0.0
)

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
    # Fallback runtime safety check to make sure keys aren't missing or invalid
    if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": None, 
            "error": "Backend Error: System could not read keys. Ensure your .env file exists and contains valid fields."
        })

    try:
        mcp_client = MultiServerMCPClient({
            "browser": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@playwright/mcp@latest", "--isolated"]
            }
        })

        # Fetch and securely flatten structural tools to resolve unhashable errors
        raw_mcp_tools = await mcp_client.get_tools()
        sanitized_tools = [t for t in raw_mcp_tools]

        # Compile agent factory using the normalized tool layout
        agent = create_agent(
            model=llm, 
            tools=sanitized_tools, 
            response_format=ExtractionOutput
        )

        full_query = f"Go to {url} immediately. Once page finishes loading, follow this prompt request: {user_prompt}"

        agent_response = await agent.ainvoke({
            "messages": [{"role": "user", "content": full_query}]
        })

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
