# main.py
import os
import datetime
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from playwright.async_api import async_playwright

# Securely pull environment properties
load_dotenv()

from langchain_openai import AzureChatOpenAI

app = FastAPI(title="Production-Ready Universal Extractor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Render paths safely relative to script execution context
current_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# Global LLM Connection Pooling
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini"), 
    api_version="2024-08-01-preview",
    temperature=0.0
)

# =====================================================================
# FLEXIBLE PYDANTIC SCHEMA
# Uses a dynamic list of string maps to accommodate any table or list structure
# =====================================================================
class StructuredDataRow(BaseModel):
    attributes: dict[str, str] = Field(
        description="Key-value pairs capturing all requested columns/fields (e.g., 'SR-No', 'Title', 'Date', 'Link')."
    )

class UniversalExtractionResponse(BaseModel):
    page_title: str = Field(description="The formal title of the source website.")
    summary: str = Field(description="High-level descriptive overview of the findings matching user intent.")
    extracted_records: list[StructuredDataRow] = Field(description="The structured rows or news items extracted from the page text.")
    data_quality_score: float = Field(description="Confidence index rating from 0.0 to 1.0 based on clarity of data source.")

# Register schema format onto the active LLM channel
structured_llm = llm.with_structured_output(UniversalExtractionResponse)

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse(request, "index.html", {"result": None, "error": None})

@app.post("/extract", response_class=HTMLResponse)
async def handle_universal_extraction(request: Request, url: str = Form(...), user_prompt: str = Form(...)):
    if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        return templates.TemplateResponse(request, "index.html", {
            "result": None, 
            "error": "Backend Error: Missing active .env credential configurations."
        })

    target_url = url if url.startswith(("http://", "https://")) else f"https://{url}"
    
    # Dynamically track today's date context for relative calculations (e.g., "latest 3 days")
    current_date_str = datetime.date.today().strftime("%B %d, %Y")

    try:
        # 1. Fire up sandboxed browser isolation layer
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            print(f"[Scraper] Loading: {target_url}")
            await page.goto(target_url, wait_until="networkidle", timeout=60000)
            
            # Extract clean page content
            page_title = await page.title()
            raw_body_text = await page.locator("body").inner_text()
            
            await browser.close()

        print("[AI Core] Commencing target text extraction pipeline...")

        # 2. Build the system core payload prompt
        system_instructions = (
            f"You are an elite data extraction agent.\n"
            f"Today's current real-world date is: {current_date_str}.\n"
            f"Source Site Scraped: '{page_title}'\n\n"
            f"--- SCRAPED BODY TEXT DATA CONTENT ---\n"
            f"{raw_body_text[:15000]}\n"
            f"--- END OF DATA CONTENT ---\n\n"
            f"User Instruction Directive: {user_prompt}\n\n"
            f"Task Guidelines:\n"
            f"1. Isolate entries/records matching the prompt rules.\n"
            f"2. If the user asks for relative dates like 'latest 3 days', use today's date ({current_date_str}) to evaluate dates found in the text.\n"
            f"3. For each target item found, extract all attributes requested by the user into the data row dictionary structure."
        )

        # 3. Synchronize structural delivery directly with Azure OpenAI channels
        structured_data: UniversalExtractionResponse = await structured_llm.ainvoke(system_instructions)

        return templates.TemplateResponse(request, "index.html", {
            "result": structured_data,
            "error": None,
            "submitted_url": url,
            "submitted_prompt": user_prompt
        })

    except Exception as e:
        import traceback
        print(f"❌ Core Application Failure Trace:\n{traceback.format_exc()}")
        return templates.TemplateResponse(request, "index.html", {
            "result": None, 
            "error": f"Process Crash: {str(e)}"
        })
