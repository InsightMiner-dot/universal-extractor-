# main.py
import os
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load variables securely from your local .env file
load_dotenv()

from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

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
    if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": None, 
            "error": "Backend Error: Missing real Azure OpenAI credentials in your .env file."
        })

    try:
        # Initializing the MCP connection context manager
        mcp_client = MultiServerMCPClient({
            "browser": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@playwright/mcp@latest", "--isolated"]
            }
        })

        # Open the browser session cleanly to prevent subprocess hang/leak crashes
        async with mcp_client.session("browser") as session:
            # Load tools natively from the active server context stream
            browser_tools = await session.get_tools()

            # Compile the agent factory using the native tools and schema definition
            agent = create_agent(
                model=llm, 
                tools=browser_tools, 
                response_format=ExtractionOutput
            )

            full_query = (
                f"1. Navigate directly to the URL: {url}\n"
                f"2. Once loaded, fulfill this extraction instruction: {user_prompt}\n"
                "3. Provide the final response strictly matching the structured layout."
            )

            # Invoke the execution graph
            agent_response = await agent.ainvoke({
                "messages": [{"role": "user", "content": full_query}]
            })

            # Safely capture the validated object output
            structured_data: ExtractionOutput = agent_response.get("structured_response")

            return templates.TemplateResponse("index.html", {
                "request": request,
                "result": structured_data,
                "error": None,
                "submitted_url": url,
                "submitted_prompt": user_prompt
            })

    except Exception as e:
        import traceback
        print(f"❌ Detailed Route Error:\n{traceback.format_exc()}")
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": None, 
            "error": f"Extraction Error: {str(e)}"
        })
