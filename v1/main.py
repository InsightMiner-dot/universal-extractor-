# main.py
import os
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables safely from .env file
load_dotenv()

from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.utils.function_calling import convert_to_openai_tool # ◄── CRITICAL IMPORT FOR THE FIX

app = FastAPI(title="Universal Web Extractor AI")

# CORS Security Allowed Domains Whitelist
origins = [
    "http://localhost:8500",
    "http://127.0.0.1:8500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Template Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# Global LLM Instantiation
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
    if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": None, 
            "error": "Backend Error: Missing real Azure OpenAI credentials in your .env file."
        })

    try:
        mcp_client = MultiServerMCPClient({
            "browser": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@playwright/mcp@latest", "--isolated"]
            }
        })

        # 1. Fetch the raw tools (which are dictionaries)
        raw_mcp_tools = await mcp_client.get_tools()
        
        # 2. FIX: Convert raw dictionaries into typed, hashable LangChain tools
        sanitized_tools = [convert_to_openai_tool(t) for t in raw_mcp_tools]

        # 3. Create agent factory using the strictly validated, hashable tools list
        agent = create_agent(
            model=llm, 
            tools=sanitized_tools, 
            response_format=ExtractionOutput
        )

        full_query = f"Go to {url} immediately. Once page finishes loading, follow this prompt request: {user_prompt}"

        # 4. Invoke agent lifecycle
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
