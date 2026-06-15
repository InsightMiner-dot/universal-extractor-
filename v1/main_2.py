import os
import asyncio
import datetime
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from langchain_openai import AzureChatOpenAI

# 1. Load environment variables from .env
load_dotenv()

# =====================================================================
# FLEXIBLE PYDANTIC SCHEMA
# =====================================================================
class StructuredDataRow(BaseModel):
    attributes: dict[str, str] = Field(
        description="Key-value pairs capturing all requested columns/fields."
    )

class UniversalExtractionResponse(BaseModel):
    page_title: str = Field(description="The formal title of the source website.")
    summary: str = Field(description="High-level descriptive overview of the findings matching user intent.")
    extracted_records: list[StructuredDataRow] = Field(description="The structured rows or news items extracted from the page text.")
    data_quality_score: float = Field(description="Confidence index rating from 0.0 to 1.0 based on clarity of data source.")

# =====================================================================
# CORE PIPELINE FUNCTIONS
# =====================================================================
async def scrape_webpage(url: str):
    """Scrapes raw text from a target URL using headless Chromium."""
    print(f"\n🌐 [Scraper] Launching browser and navigating to: {url}")
    target_url = url if url.startswith(("http://", "https://")) else f"https://{url}"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        try:
            await page.goto(target_url, wait_until="networkidle", timeout=60000)
            page_title = await page.title()
            raw_body_text = await page.locator("body").inner_text()
            return page_title, raw_body_text
        finally:
            await browser.close()

async def extract_data(page_title: str, raw_text: str, user_prompt: str):
    """Passes scraped text to Azure OpenAI for structured Pydantic extraction."""
    print("🧠 [AI Core] Processing text and extracting structured data...")
    
    # Initialize the LLM connection
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini"), 
        api_version="2024-08-01-preview",
        temperature=0.0
    )
    
    # Bind the schema format directly to the LLM
    structured_llm = llm.with_structured_output(UniversalExtractionResponse)
    
    current_date_str = datetime.date.today().strftime("%B %d, %Y")
    
    system_instructions = (
        f"You are an elite data extraction agent.\n"
        f"Today's current real-world date is: {current_date_str}.\n"
        f"Source Site Scraped: '{page_title}'\n\n"
        f"--- SCRAPED BODY TEXT DATA CONTENT ---\n"
        f"{raw_text[:15000]}\n"
        f"--- END OF DATA CONTENT ---\n\n"
        f"User Instruction Directive: {user_prompt}\n\n"
        f"Task Guidelines:\n"
        f"1. Isolate entries/records matching the prompt rules.\n"
        f"2. If the user asks for relative dates, use today's date ({current_date_str}) to evaluate dates found in the text.\n"
        f"3. For each target item found, extract all attributes requested by the user into the data row dictionary structure."
    )

    return await structured_llm.ainvoke(system_instructions)

# =====================================================================
# EXECUTION ENTRY POINT
# =====================================================================
async def main():
    # Quick credential check
    if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("❌ Error: Missing Azure OpenAI credentials. Please check your .env file.")
        return

    # Interactive terminal prompts
    print("\n" + "=" * 60)
    print("🚀 CLI Universal Web Extractor")
    print("=" * 60)
    target_url = input("👉 Enter target URL: ").strip()
    user_prompt = input("👉 Enter extraction instructions: ").strip()

    try:
        # Step 1: Execute Scrape
        title, text = await scrape_webpage(target_url)
        
        # Step 2: Execute Extraction
        result: UniversalExtractionResponse = await extract_data(title, text, user_prompt)
        
        # Step 3: Print Results cleanly to the terminal
        print("\n" + "=" * 60)
        print("✅ EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"📄 Source Title: {result.page_title}")
        print(f"📊 Quality Score: {result.data_quality_score}")
        print(f"\n📝 Summary:\n{result.summary}")
        print("\n💡 Extracted Records:")
        
        for index, record in enumerate(result.extracted_records, start=1):
            print(f"\n  [Record {index}]")
            for key, value in record.attributes.items():
                # Capitalize the key visually for cleaner reading
                print(f"    • {str(key).replace('_', ' ').title()}: {value}")
                
        print("\n" + "=" * 60 + "\n")

    except Exception as e:
        import traceback
        print(f"\n❌ Process failed:\n{traceback.format_exc()}")

if __name__ == "__main__":
    # Prevent Windows Event Loop crash issues
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
