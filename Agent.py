import os
import json
import time
import base64
from datetime import datetime, timezone
import logging
from fpdf import FPDF
import uuid
import requests
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import cv2
import numpy as np
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('youtube_converter_automation')
api_key="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"

class PlaywrightCoderAgent:
    """Agent responsible for generating Playwright automation code using OpenAI API"""
    
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        
    def generate_code(self, step_description, youtube_url=None, previous_results=None, debugging_info=None):
        """Generate Playwright code for a specific step using OpenAI API"""
        logger.info(f"Generating code for step: {step_description}")
        
        # Construct prompt based on step and context
        prompt = self._construct_prompt(step_description, youtube_url, previous_results, debugging_info)
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",  # Using GPT-4 for code generation
                messages=[
                    {"role": "system", "content": "You are an expert Playwright automation engineer. Generate precise, working Playwright code for web automation tasks. Return only the code with no explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more deterministic code generation
                max_tokens=2000
            )
            
            # Extract the code from the response
            code = response.choices[0].message.content.strip()
            
            # Ensure the code has the correct function signature
            if not "async def execute(page):" in code:
                code = "async def execute(page):\n" + code
                
            return code
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            # Fallback to template-based code if API fails
            return self._get_fallback_code(step_description, youtube_url, debugging_info)
    
    def _construct_prompt(self, step_description, youtube_url, previous_results, debugging_info):
        """Construct detailed prompt for the OpenAI API"""
        prompt = f"""
Generate Playwright code for the following automation step: "{step_description}"

The code should be written in Python using Playwright's async API and should be structured as an async function named 'execute' that takes a Playwright page object as its parameter.

The function should return a JSON-serializable object with at least:
- 'status': 'SUCCESS' or 'FAILED'
- 'message': A descriptive message about the result

Website context: The task involves a video converter website (https://video-converter.com) that has:
- An interface to open files or input URLs
- Various format options for video conversion
- A convert button to start the process

"""
        # Add YouTube URL if available
        if youtube_url:
            prompt += f"\nUse this YouTube URL for testing: {youtube_url}\n"
        
        # Add debugging context if available
        if debugging_info:
            prompt += "\nPrevious attempt failed. Consider these debugging suggestions:\n"
            for key, value in debugging_info.items():
                prompt += f"- {key}: {value}\n"
        
        # Add example code structure
        prompt += """
Example code structure:
async def execute(page):
    try:
        # Step-specific implementation
        # ...
        
        return {"status": "SUCCESS", "message": "Step completed successfully"}
    except Exception as e:
        return {"status": "FAILED", "message": f"Error: {str(e)}"}

Return ONLY the code without any additional explanations or markdown formatting.
"""
        return prompt
    
    def _get_fallback_code(self, step_description, youtube_url=None, debugging_info=None):
        """Generate fallback code when API call fails"""
        # Template-based code generation similar to the original implementation
        if "Navigate to converter" in step_description:
            code = """
async def execute(page):
    # Navigate to the video converter website
    await page.goto('https://video-converter.com/')
    
    # Wait for the main interface to load
    await page.wait_for_selector('.video-converter, input, button', timeout=10000)
    
    return {"status": "SUCCESS", "message": "Successfully navigated to the converter website"}
"""
        elif "Input YouTube URL" in step_description:
            url = youtube_url or "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            code = f"""
async def execute(page):
    # Look for URL input methods
    try:
        # Try clicking URL button if available
        url_button = await page.query_selector('text=URL')
        if url_button:
            await url_button.click()
            
        # Find and interact with URL input field
        url_input = await page.query_selector('input[type="url"], input[placeholder*="URL"], input[aria-label*="URL"]')
        if not url_input:
            # Try finding button that might open URL input dialog
            open_url_button = await page.query_selector('text="URL"')
            if open_url_button:
                await open_url_button.click()
                await page.wait_for_timeout(1000)
                url_input = await page.query_selector('input[type="url"], input[placeholder*="URL"], input[aria-label*="URL"]')
        
        if url_input:
            await url_input.fill("{url}")
            await url_input.press("Enter")
            await page.wait_for_timeout(2000)
            return {{"status": "SUCCESS", "message": "Successfully input YouTube URL"}}
        else:
            return {{"status": "FAILED", "message": "Could not find URL input field"}}
    except Exception as e:
        return {{"status": "FAILED", "message": f"Error inputting YouTube URL: {{str(e)}}"}}
"""
        elif "Attempt conversion" in step_description:
            code = """
async def execute(page):
    try:
        # Look for the convert button
        convert_button = await page.query_selector('text="Convert"')
        if convert_button:
            await convert_button.click()
            
            # Wait for conversion process to start
            await page.wait_for_timeout(5000)
            
            # Check for any error messages related to YouTube
            error_message = await page.query_selector('text="YouTube URLs not supported"')
            if error_message:
                return {"status": "FAILED", "message": "YouTube URLs not supported by this converter"}
            
            # Check for ongoing conversion process
            processing = await page.query_selector('text="Processing"')
            if processing:
                # Wait for reasonable time for conversion to progress
                await page.wait_for_timeout(10000)
                
                # Check again for errors
                error_after_wait = await page.query_selector('text/i="error"')
                if error_after_wait:
                    error_text = await page.evaluate('el => el.innerText', error_after_wait)
                    return {"status": "FAILED", "message": f"Conversion error: {error_text}"}
            
            return {"status": "SUCCESS", "message": "Conversion process started successfully"}
        else:
            return {"status": "FAILED", "message": "Could not find convert button"}
    except Exception as e:
        return {"status": "FAILED", "message": f"Error during conversion attempt: {str(e)}"}
"""
        else:
            # Generic code for unrecognized steps
            code = """
async def execute(page):
    # Generic implementation for unspecified step
    try:
        # Wait for the page to be stable
        await page.wait_for_load_state('networkidle')
        return {"status": "SUCCESS", "message": "Step completed with generic implementation"}
    except Exception as e:
        return {"status": "FAILED", "message": f"Error in generic step execution: {str(e)}"}
"""

        # Add debugging modifications if this is a retry
        if debugging_info:
            # Modify the code based on debugging suggestions
            if "selector" in debugging_info:
                code = code.replace("await page.query_selector('", f"await page.query_selector('{debugging_info['selector']}")
            if "timeout" in debugging_info:
                code = code.replace("timeout=10000", f"timeout={debugging_info['timeout']}")

        return code

class DebuggerAgent:
    """Agent responsible for analyzing failures and suggesting fixes using OpenAI API"""
    
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        
    def generate_debugging_instructions(self, step_description, error_message, screenshot_path):
        """Generate debugging instructions using OpenAI API based on error and screenshot"""
        logger.info(f"Generating debugging instructions for failed step: {step_description}")
        
        try:
            # Encode screenshot as base64 if it exists
            screenshot_data = ""
            if os.path.exists(screenshot_path):
                with open(screenshot_path, "rb") as image_file:
                    screenshot_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create prompt for debugging analysis
            prompt = f"""
Analyze the following failed automation step and suggest debugging solutions:

Step description: {step_description}
Error message: {error_message}

{"[Screenshot available for analysis]" if screenshot_data else "[No screenshot available]"}

Please provide debugging instructions in a structured JSON format containing:
1. An explanation of what likely went wrong
2. Specific suggestions for fixing the issue
3. Technical parameters that might need adjustment (selectors, timeouts, etc.)
"""

            # Call OpenAI API
            messages = [
                {"role": "system", "content": "You are an expert automation debugging assistant. Analyze web automation failures and provide specific, actionable solutions."},
                {"role": "user", "content": prompt}
            ]
            
            # Add image if available
            if screenshot_data:
                messages.append({
                    "role": "user", 
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Here is the screenshot of the failed step. Analyze what's visible and suggest fixes."
                        }
                    ]
                })
            
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",  # Using Vision model for image analysis
                messages=messages,
                temperature=0.2,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # Parse response into JSON
            debugging_json = json.loads(response.choices[0].message.content)
            
            # Ensure required fields are present
            if not isinstance(debugging_json, dict):
                debugging_json = {}
                
            # Add default fields if missing
            if "explanation" not in debugging_json:
                debugging_json["explanation"] = "Unable to determine the exact cause of failure"
            if "suggestion" not in debugging_json:
                debugging_json["suggestion"] = "Try alternative selectors and increase timeouts"
            
            # Add technical parameters if they were suggested
            if "timeout" not in debugging_json and "timeout" in error_message.lower():
                debugging_json["timeout"] = 30000  # Default increased timeout
            if "selector" not in debugging_json and "selector" in error_message.lower():
                debugging_json["selector"] = "*[data-test='input-url'], .url-input, input[type='text']"
                
            return debugging_json
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API for debugging: {str(e)}")
            # Fallback to rule-based debugging if API fails
            return self._get_fallback_debugging(step_description, error_message)
    
    def _get_fallback_debugging(self, step_description, error_message):
        """Provide fallback debugging information when API call fails"""
        debugging_info = {}
        
        # Analyze common failure patterns
        if "timeout" in error_message.lower():
            debugging_info = {
                "explanation": "The operation timed out, suggesting the element wasn't found in time",
                "suggestion": "Increase timeout duration and use more reliable selectors",
                "timeout": 30000,  # Increase timeout to 30 seconds
            }
        elif "selector" in error_message.lower():
            debugging_info = {
                "explanation": "The specified selector wasn't found on the page",
                "suggestion": "Use more generic selectors or alternative approaches",
                "selector": "*[data-test='input-url'], .url-input, input[type='text']"  # More generic selector
            }
        elif "youtube" in error_message.lower():
            debugging_info = {
                "explanation": "This converter may not support YouTube URLs directly",
                "suggestion": "Try a different approach like downloading first, or try a different converter",
                "alternative": "Try uploading a local video file instead"
            }
        else:
            debugging_info = {
                "explanation": "Unknown error occurred during execution",
                "suggestion": "Add more logging and implement more robust error handling",
                "retry": True
            }
            
        return debugging_info

class ScreenshotAnalyzerAgent:
    """Agent responsible for analyzing screenshots to determine UI state using OpenAI Vision API"""
    
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    
    def analyze_screenshot(self, screenshot_path, step_description):
        """Analyze screenshot using OpenAI Vision API to determine if step was successful"""
        logger.info(f"Analyzing screenshot for step: {step_description}")
        
        # Check if screenshot exists
        if not os.path.exists(screenshot_path):
            return {
                "status": "ERROR", 
                "message": "Screenshot file does not exist",
                "confidence": 0.0
            }
        
        try:
            # Encode screenshot as base64
            with open(screenshot_path, "rb") as image_file:
                screenshot_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Construct prompt for the OpenAI API
            prompt = f"""
Analyze this screenshot of a video converter website interface after the following step: "{step_description}"

Determine whether the step appears to have been successful or failed based on visual evidence.
Identify key UI elements visible in the screenshot.
Look for any error messages, success indicators, or expected UI states.

For this specific step, you should check for:
"""
            
            # Add step-specific guidance
            if "Navigate to converter" in step_description:
                prompt += """
- The main interface of the video converter is visible
- There are options to upload or input URLs
- The convert button is present
"""
            elif "Input YouTube URL" in step_description:
                prompt += """
- A YouTube URL has been entered into an input field
- The UI shows the URL is recognized/accepted
- The convert option is enabled/available
"""
            elif "Attempt conversion" in step_description:
                prompt += """
- Look for processing indicators or progress bars
- Check for error messages specifically about YouTube URLs
- Look for success indicators or download options
"""
                
            prompt += """
Provide your analysis in a structured JSON format with the following fields:
1. "status": either "SUCCESS", "FAILED", or "UNCLEAR"
2. "ui_elements_detected": array of key UI elements visible
3. "confidence": a number between 0 and 1 indicating your confidence level
4. "explanation": detailed explanation of your assessment
"""

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert UI analyzer for web automation testing. Analyze screenshots and determine if automated steps succeeded or failed."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{screenshot_data}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                temperature=0.2,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            analysis_result = json.loads(response.choices[0].message.content)
            
            # Ensure required fields exist
            if "status" not in analysis_result:
                analysis_result["status"] = "UNCLEAR"
            if "ui_elements_detected" not in analysis_result:
                analysis_result["ui_elements_detected"] = []
            if "confidence" not in analysis_result:
                analysis_result["confidence"] = 0.5
            if "explanation" not in analysis_result:
                analysis_result["explanation"] = "Analysis was inconclusive"
                
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing screenshot with OpenAI API: {str(e)}")
            # Fallback to basic analysis
            return self._fallback_analysis(screenshot_path, step_description)
    
    def _fallback_analysis(self, screenshot_path, step_description):
        """Provide fallback analysis when API call fails"""
        # Basic image analysis without AI
        try:
            # Load the image
            image = cv2.imread(screenshot_path)
            if image is None:
                return {"status": "ERROR", "message": "Could not load screenshot for analysis", "confidence": 0.0}
            
            # Very basic analysis based on image properties
            analysis_result = {
                "status": "SUCCESS",  # Default to success unless we find evidence otherwise
                "ui_elements_detected": ["page loaded"],
                "confidence": 0.6,
                "explanation": "Basic analysis detected a loaded page. Detailed AI analysis failed."
            }
            
            # Look for common error colors (lots of red might indicate errors)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            red_lower = np.array([0, 100, 100])
            red_upper = np.array([10, 255, 255])
            red_mask = cv2.inRange(hsv, red_lower, red_upper)
            red_percentage = np.sum(red_mask > 0) / (image.shape[0] * image.shape[1])
            
            if red_percentage > 0.05:  # If more than 5% of the image has red
                analysis_result["status"] = "FAILED"
                analysis_result["explanation"] = "Image contains significant amount of red color, which might indicate errors"
                analysis_result["confidence"] = 0.7
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in fallback screenshot analysis: {str(e)}")
            return {
                "status": "UNCLEAR",
                "ui_elements_detected": [],
                "confidence": 0.3,
                "explanation": f"Both API and fallback analysis failed: {str(e)}"
            }

class YouTubeConverterAutomation:
    """Main automation system that coordinates the agents and workflow"""
    
    def __init__(self, youtube_url, api_key=None):
        self.youtube_url = youtube_url
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # Initialize agents with OpenAI API key
        self.playwright_coder = PlaywrightCoderAgent(api_key=self.api_key)
        self.debugger = DebuggerAgent(api_key=self.api_key)
        self.screenshot_analyzer = ScreenshotAnalyzerAgent(api_key=self.api_key)
        
        # Create output directory
        self.output_dir = os.path.join(os.getcwd(), "output_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Test run info
        self.test_id = f"youtube-conversion-{uuid.uuid4().hex[:8]}"
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.end_time = None
        self.steps = []
        
    def run_automation(self):
        """Run the complete automation workflow"""
        logger.info(f"Starting automation with test ID: {self.test_id}")
        
        # Define the steps for the workflow
        workflow_steps = [
            "Navigate to converter",
            "Input YouTube URL",
            "Attempt conversion"
        ]
        
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=False)
            context = browser.new_context()
            page = context.new_page()
            
            for step_idx, step_description in enumerate(workflow_steps):
                step_result = self._execute_step(page, step_idx + 1, step_description)
                self.steps.append(step_result)
                
                # Take a screenshot after every step execution
                screenshot_path = os.path.join(self.output_dir, f"step{step_idx + 1}_final.png")
                try:
                    # Wait briefly for any animations or changes to complete
                    page.wait_for_timeout(2000)  
                    page.screenshot(path=screenshot_path)
                    step_result["final_screenshot"] = screenshot_path
                    logger.info(f"Captured final screenshot for step {step_idx + 1} at {screenshot_path}")
                except Exception as e:
                    logger.error(f"Error capturing final screenshot for step {step_idx + 1}: {str(e)}")
                
                # Stop execution if a step fails after all retries
                if step_result["status"] == "FAILED":
                    logger.warning(f"Step '{step_description}' failed after all retry attempts. Stopping workflow.")
                    break
            
            browser.close()
        
        # Record end time
        self.end_time = datetime.now(timezone.utc).isoformat()
        
        # Generate final report
        self._generate_final_report()
        
        return self.steps
    
    def _execute_step(self, page, step_number, step_description, retry_count=0, debugging_info=None):
        """Execute a single step with retry mechanism"""
        if retry_count >= 3:
            logger.error(f"Maximum retry attempts reached for step: {step_description}")
            return {
                "step": step_description,
                "status": "FAILED",
                "error": "Maximum retry attempts reached",
                "screenshot": f"step{step_number}_retry{retry_count}.png"
            }
        
        logger.info(f"Executing step {step_number}: {step_description}" + 
                (f" (Retry {retry_count+1})" if retry_count > 0 else ""))
        
        # 1. Generate Playwright code
        code = self.playwright_coder.generate_code(
            step_description, 
            youtube_url=self.youtube_url,
            previous_results=self.steps if self.steps else None,
            debugging_info=debugging_info
        )
        code_file_path = os.path.join(self.output_dir, f"step{step_number}_code{'_retry'+str(retry_count) if retry_count else ''}.py")
        
        with open(code_file_path, "w") as f:
            f.write(code)
        
        # 2. Run the generated code - DIRECT EXECUTION APPROACH
        try:
            # Make sure imports are available in the execution context
            import asyncio
            from playwright.async_api import async_playwright
            
            # Prepare execute function
            namespace = {}
            exec(code, namespace)
            
            # Define wrapper to run async code
            async def run_async_code():
                async with async_playwright() as p:
                    browser = await p.chromium.launch()
                    context = await browser.new_context()
                    page_for_step = await context.new_page()
                    
                    # Execute the code
                    result = await namespace["execute"](page_for_step)
                    
                    # Take screenshot
                    screenshot_path = os.path.join(self.output_dir, f"step{step_number}{'_retry'+str(retry_count) if retry_count else ''}.png")
                    await page_for_step.screenshot(path=screenshot_path)
                    
                    await browser.close()
                    return result, screenshot_path
            
            # Run async code
            result, screenshot_path = asyncio.run(run_async_code())
            
            # 5. Analyze the screenshot
            analysis_result = self.screenshot_analyzer.analyze_screenshot(screenshot_path, step_description)
            
            # 6. Save result
            result_path = os.path.join(self.output_dir, f"step{step_number}_result{'_retry'+str(retry_count) if retry_count else ''}.json")
            with open(result_path, "w") as f:
                json.dump({
                    "execution_result": result,
                    "screenshot_analysis": analysis_result
                }, f, indent=2)
            
            # 7. Check if successful
            step_status = "SUCCESS" if result.get("status") == "SUCCESS" and analysis_result.get("status") == "SUCCESS" else "FAILED"
            
            if step_status == "FAILED":
                # 8. Generate debugging instructions
                error_message = result.get("message", "Unknown error")
                debugging_info = self.debugger.generate_debugging_instructions(step_description, error_message, screenshot_path)
                
                # Save debugging info
                debug_path = os.path.join(self.output_dir, f"step{step_number}_debug{'_retry'+str(retry_count) if retry_count else ''}.json")
                with open(debug_path, "w") as f:
                    json.dump(debugging_info, f, indent=2)
                
                # 9. Retry the step
                logger.info(f"Retrying step {step_number} with debugging information")
                return self._execute_step(page, step_number, step_description, retry_count + 1, debugging_info)
            
            return {
                "step": step_description,
                "status": step_status,
                "screenshot": os.path.basename(screenshot_path)
            }
            
        except Exception as e:
            logger.error(f"Error executing step {step_number}: {str(e)}")
            
            # Save error screenshot if page is still available
            error_screenshot_path = os.path.join(self.output_dir, f"step{step_number}_error{'_retry'+str(retry_count) if retry_count else ''}.png")
            try:
                page.screenshot(path=error_screenshot_path)
            except:
                logger.error("Could not capture error screenshot")
            
            # Generate debugging info and retry
            debugging_info = self.debugger.generate_debugging_instructions(step_description, str(e), error_screenshot_path)
            
            # Save debugging info
            debug_path = os.path.join(self.output_dir, f"step{step_number}_debug{'_retry'+str(retry_count) if retry_count else ''}.json")
            with open(debug_path, "w") as f:
                json.dump(debugging_info, f, indent=2)
            
            # Retry the step
            return self._execute_step(page, step_number, step_description, retry_count + 1, debugging_info)
        
    def _generate_final_report(self):
        """Generate a final PDF report with all results and screenshots"""
        logger.info("Generating final report")
        
        # Create report data structure
        report_data = {
            "testId": self.test_id,
            "status": "SUCCESS" if all(step["status"] == "SUCCESS" for step in self.steps) else "FAILED",
            "startTime": self.start_time,
            "endTime": self.end_time,
            "steps": self.steps,
            "aiAnalysis": {
                "explanation": self._generate_overall_analysis(),
                "recommendation": self._generate_recommendations(),
                "confidence": 0.95
            }
        }
        
        # Save JSON report
        json_report_path = os.path.join(self.output_dir, "final_report.json")
        with open(json_report_path, "w") as f:
            json.dump(report_data, f, indent=2)
        
        # Create PDF report
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"YouTube Converter Test Report: {self.test_id}", ln=True)
        
        # Test info
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Status: {report_data['status']}", ln=True)
        pdf.cell(0, 10, f"Start Time: {report_data['startTime']}", ln=True)
        pdf.cell(0, 10, f"End Time: {report_data['endTime']}", ln=True)
        
        # Steps summary
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Steps Summary", ln=True)
        
        pdf.set_font("Arial", "", 12)
        for step in self.steps:
            pdf.cell(0, 10, f"Step: {step['step']}", ln=True)
            pdf.cell(0, 10, f"Status: {step['status']}", ln=True)
            
            # Add step details if available (like error messages)
            if "error" in step:
                pdf.multi_cell(0, 10, f"Error: {step['error']}")
            
            # Add screenshot from each step attempt
            screenshot_path = os.path.join(self.output_dir, step.get('screenshot', ''))
            if os.path.exists(screenshot_path):
                pdf.cell(0, 10, f"Step Execution Screenshot:", ln=True)
                pdf.image(screenshot_path, x=10, w=180)
            
            # Add the final screenshot after the step
            if "final_screenshot" in step and os.path.exists(step["final_screenshot"]):
                pdf.cell(0, 10, f"Final State After Step:", ln=True)
                pdf.image(step["final_screenshot"], x=10, w=180)
            
            # Add a page between steps for better organization
            pdf.add_page()
        
        # AI Analysis
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "AI Analysis", ln=True)
        
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, f"Explanation: {report_data['aiAnalysis']['explanation']}")
        pdf.multi_cell(0, 10, f"Recommendation: {report_data['aiAnalysis']['recommendation']}")
        pdf.cell(0, 10, f"Confidence: {report_data['aiAnalysis']['confidence']}", ln=True)
        
        # Add summary of screenshots
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "All Screenshots Summary", ln=True)
        
        # Display all final screenshots in a summary page
        for idx, step in enumerate(self.steps):
            if "final_screenshot" in step and os.path.exists(step["final_screenshot"]):
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, f"Step {idx+1}: {step['step']}", ln=True)
                pdf.set_font("Arial", "", 10)
                pdf.cell(0, 5, f"Status: {step['status']}", ln=True)
                pdf.image(step["final_screenshot"], x=10, w=180)
                # Add some space between images
                pdf.cell(0, 10, "", ln=True)
        
        # Save PDF
        pdf_path = os.path.join(self.output_dir, "final_report.pdf")
        pdf.output(pdf_path)
        
        logger.info(f"Final report generated at: {pdf_path}")
        return pdf_path


# Main execution code
if __name__ == "__main__":
    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not found.")
        api_key = input("Please enter your OpenAI API key (or press Enter to run without AI features): ").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            print("Running with limited functionality (no AI-powered analysis).")
    
    youtube_url = input("Enter YouTube URL to convert: ")
    
    try:
        # Setup Playwright (ensure browser binaries are installed)
        os.system("playwright install chromium")
        
        print("Starting YouTube Converter Automation...")
        print("Screenshots will be taken after each step and included in the final PDF report.")
        
        # Run automation
        automation = YouTubeConverterAutomation(youtube_url, api_key=api_key)
        results = automation.run_automation()
        
        print("\nAutomation completed.")
        print(f"Results saved to: {automation.output_dir}")
        print(f"Final report: {os.path.join(automation.output_dir, 'final_report.pdf')}")
        print(f"The PDF report includes screenshots from all steps of the automation process.")
        
        # Open report if possible
        if os.path.exists(os.path.join(automation.output_dir, 'final_report.pdf')):
            print("Opening final report...")
            try:
                import webbrowser
                webbrowser.open(os.path.join(automation.output_dir, 'final_report.pdf'))
            except:
                print("Could not automatically open the report.")
    
    except Exception as e:
        print(f"An error occurred during automation: {str(e)}")
        import traceback
        traceback.print_exc()
