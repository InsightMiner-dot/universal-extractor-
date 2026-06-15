# main.py
import os
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from playwright.async_api import async_playwright  # ◄── Native High-Speed Browser Engine

# Load your local .env file securely
load_dotenv()

from langchain_openai import AzureChatOpenAI

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

# Bind the structured output schema directly to the global LLM connection
structured_llm = llm.with_structured_output(ExtractionOutput)

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse(request, "index.html", {"result": None, "error": None})

@app.post("/extract", response_class=HTMLResponse)
async def handle_extraction(request: Request, url: str = Form(...), user_prompt: str = Form(...)):
    if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        return templates.TemplateResponse(request, "index.html", {
            "result": None, 
            "error": "Backend Error: Missing Azure OpenAI credentials. Please check your .env file."
        })

    # Ensure URLs have a proper protocol prefix
    target_url = url if url.startswith(("http://", "https://")) else f"https://{url}"

    try:
        # 1. Start the single-instance browser automation lifecycle
        async with async_playwright() as p:
            # Change headless=True to headless=False if you want to see it open smoothly once
            browser = await p.chromium.launch(headless=True) 
            page = await browser.new_page()
            
            print(f"[Browser] Navigating directly to: {target_url}")
            # Navigate and wait until the network goes completely quiet (ajax/fonts loaded)
            await page.goto(target_url, wait_until="networkidle", timeout=45000)
            
            # Extract high-level metadata content safely
            page_title = await page.title()
            page_text = await page.locator("body").inner_text()
            
            # Clean up the browser instance immediately
            await browser.close()

        print("[AI Engine] Processing web text content via Structured Extraction...")
        
        # 2. Package the text content and user request into a targeted prompt
        system_prompt = (
            f"You are an expert data extraction assistant.\n"
            f"We have successfully scraped the webpage titled: '{page_title}'.\n\n"
            f"Here is the raw text content from the webpage:\n"
            f"--- START CONTENT ---\n{page_text[:12000]}\n--- END CONTENT ---\n\n"
            f"Your instruction: {user_prompt}\n"
            f"Analyze the content above and fill out the required response format fields accurately."
        )

        # 3. Invoke the structured LLM directly (guarantees a perfect Pydantic object)
        structured_data: ExtractionOutput = await structured_llm.ainvoke(system_prompt)

        return templates.TemplateResponse(request, "index.html", {
            "result": structured_data,
            "error": None,
            "submitted_url": url,
            "submitted_prompt": user_prompt
        })

    except Exception as e:
        import traceback
        print(f"❌ Extraction Process Error:\n{traceback.format_exc()}")
        return templates.TemplateResponse(request, "index.html", {
            "result": None, 
            "error": f"Extraction Error: {str(e)}"
        })
