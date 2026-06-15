# main.py
import os
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load your local .env file securely
load_dotenv()

from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

app = FastAPI(title="Local Browser Extractor AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
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
    # FIXED: request is now passed as the first positional argument
    return templates.TemplateResponse(request, "index.html", {"result": None, "error": None})

@app.post("/extract", response_class=HTMLResponse)
async def handle_extraction(request: Request, url: str = Form(...), user_prompt: str = Form(...)):
    if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        # FIXED: request is passed first
        return templates.TemplateResponse(request, "index.html", {
            "result": None, 
            "error": "Backend Error: Missing Azure OpenAI credentials. Please check your .env file."
        })

    try:
        mcp_client = MultiServerMCPClient({
            "browser": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@playwright/mcp@latest", "--isolated"]
            }
        })

        # Fetch the tools directly from the global client object
        browser_tools = await mcp_client.get_tools()

        # Create a stable ReAct agent using LangGraph
        agent = create_react_agent(llm, tools=browser_tools)

        # Give the agent strict instructions to output raw JSON matching your schema
        system_instruction = (
            f"You are a web extraction assistant. Navigate to {url}. "
            f"Fulfill this task: {user_prompt}\n\n"
            "Once you have the data, you MUST return your final response as a raw JSON object "
            "matching this exact structure, with no markdown formatting or extra text:\n"
            "{\n"
            '  "page_title": "string",\n'
            '  "scraped_summary": "string",\n'
            '  "key_insights": ["string1", "string2"],\n'
            '  "data_quality_score": 0.95\n'
            "}"
        )

        agent_response = await agent.ainvoke({
            "messages": [{"role": "user", "content": system_instruction}]
        })

        # Extract the final JSON string from the AI's last message
        raw_text_output = agent_response["messages"][-1].content
        
        # Strip markdown code blocks to ensure safe parsing
        cleaned_json_string = raw_text_output.replace("```json", "").replace("```", "").strip()
        
        # Parse into your structured Pydantic object
        structured_data = ExtractionOutput.model_validate_json(cleaned_json_string)

        # Close connections cleanly
        await mcp_client.close()

        # FIXED: request is passed first
        return templates.TemplateResponse(request, "index.html", {
            "result": structured_data,
            "error": None,
            "submitted_url": url,
            "submitted_prompt": user_prompt
        })

    except Exception as e:
        import traceback
        print(f"❌ Detailed Console Error:\n{traceback.format_exc()}")
        
        # FIXED: request is passed first
        return templates.TemplateResponse(request, "index.html", {
            "result": None, 
            "error": f"Extraction Error: {str(e)}"
        })
