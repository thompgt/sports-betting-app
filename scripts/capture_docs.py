import asyncio
import os
import subprocess
import time
from playwright.async_api import async_playwright

async def capture_dashboard():
    # 1. Start Streamlit in the background
    print("Launching Streamlit server...")
    process = subprocess.Popen(
        ["streamlit", "run", "app/ui/dashboard.py", "--server.port", "8501", "--server.headless", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(10)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(viewport={'width': 1280, 'height': 800})
        page = await context.new_page()
        
        print("Navigating to dashboard...")
        try:
            await page.goto("http://localhost:8501", timeout=90000)
            # Wait for content to load
            print("Waiting for dashboard elements...")
            await asyncio.sleep(15) # Wait for Streamlit to render completely
            
            output_path = "docs/assets/dashboard_preview.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            await page.screenshot(path=output_path)
            print(f"Screenshot saved to {output_path}")
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
        finally:
            await browser.close()
            process.terminate()

if __name__ == "__main__":
    asyncio.run(capture_dashboard())
