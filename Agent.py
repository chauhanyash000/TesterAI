import os
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, TypedDict, Annotated, Literal, Union

# LangGraph and LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage
from pydantic import BaseModel, Field
import langgraph.graph as lg
from langgraph.graph.graph import END

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, UnexpectedAlertPresentException

# For screenshots and reporting
import matplotlib.pyplot as plt
from PIL import Image

from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

class AgentState(TypedDict):
    """State for the agent workflow"""
    messages: List[Any]  # Messages exchanged in the conversation
    website_data: Dict[str, Any]  # Current data about the website state
    test_results: List[Dict[str, Any]]  # Results of each test case
    current_step: int  # Current step in the test plan
    error: Optional[str]  # Current error state, if any
    done: bool  # Whether the agent has completed its tasks

# Define tools for the agent to interact with the website
class FindElement(BaseModel):
    """Find an element on the webpage."""
    selector_type: Literal["id", "class", "name", "xpath", "css", "tag", "link_text"] = Field(
        description="Type of selector to use (id, class, name, xpath, css, tag, link_text)"
    )
    selector_value: str = Field(
        description="The value of the selector to search for"
    )

    def run(self, driver_setup):
        return driver_setup.find_element(self.selector_type, self.selector_value)

class ClickElement(BaseModel):
    """Click on an element on the webpage."""
    selector_type: Literal["id", "class", "name", "xpath", "css", "tag", "link_text"] = Field(
        description="Type of selector to use"
    )
    selector_value: str = Field(
        description="The value of the selector to search for"
    )

    def run(self, driver_setup):
        return driver_setup.click_element(self.selector_type, self.selector_value)

class TypeText(BaseModel):
    """Type text into an input field."""
    selector_type: Literal["id", "class", "name", "xpath", "css", "tag", "link_text"] = Field(
        description="Type of selector to use"
    )
    selector_value: str = Field(
        description="The value of the selector to search for"
    )
    text: str = Field(
        description="The text to type into the field"
    )

    def run(self, driver_setup):
        return driver_setup.type_text(self.selector_type, self.selector_value, self.text)

class TakeScreenshot(BaseModel):
    """Take a screenshot of the current page."""
    description: str = Field(
        description="Description of what the screenshot captures"
    )

    def run(self, driver_setup):
        return driver_setup.take_screenshot(self.description)

class NavigateTo(BaseModel):
    """Navigate to a specific URL."""
    url: str = Field(
        description="The URL to navigate to"
    )

    def run(self, driver_setup):
        return driver_setup.navigate_to(self.url)

class WaitFor(BaseModel):
    """Wait for an element to appear on the page."""
    selector_type: Literal["id", "class", "name", "xpath", "css", "tag", "link_text"] = Field(
        description="Type of selector to use"
    )
    selector_value: str = Field(
        description="The value of the selector to search for"
    )
    timeout: int = Field(
        description="Maximum time to wait in seconds",
        default=10
    )

    def run(self, driver_setup):
        return driver_setup.wait_for(self.selector_type, self.selector_value, self.timeout)

class GetPageSource(BaseModel):
    """Get the HTML source of the current page."""
    
    def run(self, driver_setup):
        return driver_setup.get_page_source()

# New tool for handling alert boxes
class HandleAlert(BaseModel):
    """Handle JavaScript alert, confirm, or prompt dialogs."""
    action: Literal["accept", "dismiss", "get_text", "send_text"] = Field(
        description="Action to perform on the alert: accept (OK), dismiss (Cancel), get_text (read alert message), send_text (for prompts)"
    )
    text: Optional[str] = Field(
        description="Text to send to the prompt dialog (only used if action is 'send_text')",
        default=None
    )

    def run(self, driver_setup):
        return driver_setup.handle_alert(self.action, self.text)

class CheckAlertPresent(BaseModel):
    """Check if an alert is present on the page."""
    
    def run(self, driver_setup):
        return driver_setup.check_alert_present()

class DriverSetup:
    def __init__(self):
        """Initialize the WebDriver."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode (no GUI)
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)
        self.screenshots_dir = "screenshots"
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
    def find_element(self, selector_type: str, selector_value: str) -> Dict[str, Any]:
        """Find an element on the page."""
        try:
            # First check if an alert is present
            if self._is_alert_present():
                return {"success": False, "error": "Alert is present. Handle the alert first before finding elements."}
            
            selector = self._get_selector(selector_type, selector_value)
            element = self.wait.until(EC.presence_of_element_located(selector))
            
            # Get basic info about the element
            tag_name = element.tag_name
            text = element.text
            is_displayed = element.is_displayed()
            is_enabled = element.is_enabled()
            
            return {
                "success": True,
                "element_found": True,
                "tag_name": tag_name,
                "text": text,
                "is_displayed": is_displayed,
                "is_enabled": is_enabled
            }
        except TimeoutException:
            return {"success": False, "error": f"Element not found: {selector_type}={selector_value}"}
        except UnexpectedAlertPresentException:
            # Alert appeared - inform the agent to handle it
            return {"success": False, "error": "Unexpected alert present. Use HandleAlert tool to interact with it."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def click_element(self, selector_type: str, selector_value: str) -> Dict[str, Any]:
        """Click on an element."""
        try:
            # First check if an alert is present
            if self._is_alert_present():
                return {"success": False, "error": "Alert is present. Handle the alert first before clicking elements."}
            
            selector = self._get_selector(selector_type, selector_value)
            element = self.wait.until(EC.element_to_be_clickable(selector))
            element.click()
            time.sleep(1)  # Wait for any animations or state changes
            
            # After clicking, check if an alert appeared
            if self._is_alert_present():
                alert_text = self.driver.switch_to.alert.text
                return {
                    "success": True, 
                    "message": f"Clicked on {selector_type}={selector_value}",
                    "alert_appeared": True,
                    "alert_text": alert_text
                }
            
            return {"success": True, "message": f"Clicked on {selector_type}={selector_value}", "alert_appeared": False}
        except TimeoutException:
            return {"success": False, "error": f"Element not clickable: {selector_type}={selector_value}"}
        except UnexpectedAlertPresentException:
            # Alert appeared during the operation
            try:
                alert_text = self.driver.switch_to.alert.text
                return {
                    "success": True, 
                    "message": f"Clicked on {selector_type}={selector_value}",
                    "alert_appeared": True,
                    "alert_text": alert_text
                }
            except:
                return {
                    "success": True, 
                    "message": f"Clicked on {selector_type}={selector_value}",
                    "alert_appeared": True,
                    "alert_text": "Unable to get alert text"
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def type_text(self, selector_type: str, selector_value: str, text: str) -> Dict[str, Any]:
        """Type text into an input field."""
        try:
            # First check if an alert is present
            if self._is_alert_present():
                return {"success": False, "error": "Alert is present. Handle the alert first before typing text."}
            
            selector = self._get_selector(selector_type, selector_value)
            element = self.wait.until(EC.element_to_be_clickable(selector))
            element.clear()
            element.send_keys(text)
            return {"success": True, "message": f"Typed '{text}' into {selector_type}={selector_value}"}
        except TimeoutException:
            return {"success": False, "error": f"Element not available for typing: {selector_type}={selector_value}"}
        except UnexpectedAlertPresentException:
            return {"success": False, "error": "Unexpected alert present. Use HandleAlert tool to interact with it."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def take_screenshot(self, description: str) -> Dict[str, Any]:
        """Take a screenshot of the current page."""
        try:
            # First check if an alert is present - alerts can block screenshot in some browsers
            if self._is_alert_present():
                # Take a screenshot of alert state (note: alerts might not be visible in headless mode)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.screenshots_dir}/alert_screenshot_{timestamp}.png"
                self.driver.save_screenshot(filename)
                alert_text = self.driver.switch_to.alert.text
                return {
                    "success": True,
                    "screenshot_path": filename,
                    "description": f"{description} (Alert present: '{alert_text}')"
                }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.screenshots_dir}/screenshot_{timestamp}.png"
            self.driver.save_screenshot(filename)
            return {
                "success": True,
                "screenshot_path": filename,
                "description": description
            }
        except UnexpectedAlertPresentException:
            # Try again but handle the alert first
            alert_text = "Unknown alert text"
            try:
                alert_text = self.driver.switch_to.alert.text
            except:
                pass
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.screenshots_dir}/alert_screenshot_{timestamp}.png"
            try:
                self.driver.save_screenshot(filename)
                return {
                    "success": True,
                    "screenshot_path": filename,
                    "description": f"{description} (Alert present: '{alert_text}')"
                }
            except:
                return {"success": False, "error": f"Failed to take screenshot due to alert: {alert_text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def navigate_to(self, url: str) -> Dict[str, Any]:
        """Navigate to a specific URL."""
        try:
            # First check if an alert is present
            if self._is_alert_present():
                return {"success": False, "error": "Alert is present. Handle the alert first before navigating."}
            
            self.driver.get(url)
            time.sleep(2)  # Wait for page to load
            return {"success": True, "current_url": self.driver.current_url}
        except UnexpectedAlertPresentException:
            return {"success": False, "error": "Unexpected alert present. Use HandleAlert tool to interact with it."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def wait_for(self, selector_type: str, selector_value: str, timeout: int) -> Dict[str, Any]:
        """Wait for an element to appear on the page."""
        try:
            # First check if an alert is present
            if self._is_alert_present():
                return {"success": False, "error": "Alert is present. Handle the alert first before waiting for elements."}
            
            selector = self._get_selector(selector_type, selector_value)
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located(selector)
            )
            return {"success": True, "message": f"Element {selector_type}={selector_value} found after waiting"}
        except TimeoutException:
            return {"success": False, "error": f"Timed out waiting for element: {selector_type}={selector_value}"}
        except UnexpectedAlertPresentException:
            return {"success": False, "error": "Unexpected alert present. Use HandleAlert tool to interact with it."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_page_source(self) -> Dict[str, Any]:
        """Get the HTML source of the current page."""
        try:
            # First check if an alert is present
            if self._is_alert_present():
                return {
                    "success": False, 
                    "error": "Alert is present. Handle the alert first before getting page source."
                }
            
            source = self.driver.page_source
            return {
                "success": True,
                "source_length": len(source),
                "title": self.driver.title,
                "current_url": self.driver.current_url
            }
        except UnexpectedAlertPresentException:
            return {"success": False, "error": "Unexpected alert present. Use HandleAlert tool to interact with it."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def handle_alert(self, action: str, text: Optional[str] = None) -> Dict[str, Any]:
        """Handle JavaScript alert boxes."""
        try:
            # Check if alert is present
            if not self._is_alert_present():
                return {"success": False, "error": "No alert is present"}
            
            # Switch to the alert
            alert = self.driver.switch_to.alert
            alert_text = alert.text
            
            # Perform the requested action
            if action == "accept":
                alert.accept()
                return {"success": True, "message": "Alert accepted", "alert_text": alert_text}
            elif action == "dismiss":
                alert.dismiss()
                return {"success": True, "message": "Alert dismissed", "alert_text": alert_text}
            elif action == "get_text":
                return {"success": True, "message": "Alert text retrieved", "alert_text": alert_text}
            elif action == "send_text":
                if text is None:
                    return {"success": False, "error": "Text parameter required for send_text action"}
                alert.send_keys(text)
                alert.accept()
                return {"success": True, "message": f"Sent text '{text}' to prompt", "alert_text": alert_text}
            else:
                return {"success": False, "error": f"Unknown alert action: {action}"}
        except Exception as e:
            return {"success": False, "error": f"Error handling alert: {str(e)}"}

    def check_alert_present(self) -> Dict[str, Any]:
        """Check if an alert is present and return its details."""
        is_present = self._is_alert_present()
        
        if is_present:
            try:
                alert_text = self.driver.switch_to.alert.text
                return {
                    "success": True,
                    "alert_present": True,
                    "alert_text": alert_text
                }
            except:
                return {
                    "success": True,
                    "alert_present": True,
                    "alert_text": "Unable to get alert text"
                }
        else:
            return {
                "success": True,
                "alert_present": False
            }

    def _is_alert_present(self) -> bool:
        """Helper method to check if an alert is present."""
        try:
            WebDriverWait(self.driver, 0.5).until(EC.alert_is_present())
            return True
        except:
            return False

    def _get_selector(self, selector_type: str, selector_value: str) -> Tuple:
        """Convert selector type and value to Selenium By selector."""
        selector_map = {
            "id": By.ID,
            "class": By.CLASS_NAME,
            "name": By.NAME,
            "xpath": By.XPATH,
            "css": By.CSS_SELECTOR,
            "tag": By.TAG_NAME,
            "link_text": By.LINK_TEXT
        }
        return (selector_map[selector_type], selector_value)

    def close(self):
        """Close the browser."""
        if self.driver:
            self.driver.quit()

# Define the agent using LangGraph
class VideoConverterTestingAgent:
    def __init__(self, openai_api_key: str):
        """Initialize the agent with the necessary components."""
        self.openai_api_key = openai_api_key
        self.driver_setup = DriverSetup()
        self.report_path = "testing_report.json"
        
        # Initialize LLM with direct tools argument
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=openai_api_key
        )
        
        # Define tools (including new alert handling tools)
        self.tools = [
            FindElement, 
            ClickElement, 
            TypeText, 
            TakeScreenshot, 
            NavigateTo, 
            WaitFor, 
            GetPageSource,
            HandleAlert,
            CheckAlertPresent
        ]
        
        # Convert tools to OpenAI functions
        self.functions = []
        for tool_cls in self.tools:
            # Create a schema for each tool
            schema = {
                "type": "function",
                "function": {
                    "name": tool_cls.__name__,
                    "description": tool_cls.__doc__,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            
            # Extract properties from the model's fields
            for field_name, field in tool_cls.model_fields.items():
                schema["function"]["parameters"]["properties"][field_name] = {
                    "type": "string",  # Simplify to string for all types
                    "description": field.description
                }
                if field.is_required:
                    if "required" not in schema["function"]["parameters"]:
                        schema["function"]["parameters"]["required"] = []
                    schema["function"]["parameters"]["required"].append(field_name)
                    
            self.functions.append(schema)
        
        # Create the agent workflow graph
        self.workflow = self._create_workflow()
        
    def _create_workflow(self):
        """Create the LangGraph workflow."""
        # Define node functions
        
        def agent(state: AgentState) -> AgentState:
            """Agent node: Decides what to do next based on the current state."""
            messages = state["messages"]
            
            # Create a prompt with structured format instructions
            instruction_content = """
            You are a testing agent for a video converter website. Your task is to interact with the website,
            test its functionality, and report on the results.
            
            Available actions:
            1. FindElement - Find an element on the webpage
            2. ClickElement - Click on a button or link
            3. TypeText - Type text into an input field
            4. TakeScreenshot - Take a screenshot
            5. NavigateTo - Navigate to a URL
            6. WaitFor - Wait for an element to appear
            7. GetPageSource - Get the HTML source
            8. HandleAlert - Interact with JavaScript alert boxes
            9. CheckAlertPresent - Check if an alert is present
            
            IMPORTANT: This website may display JavaScript alert boxes when performing certain actions.
            Always check for alerts and handle them appropriately. If an action results in an alert box,
            you should:
            1. Check if an alert is present using CheckAlertPresent
            2. If present, use HandleAlert to accept, dismiss, or interact with the alert 
            3. Then continue with your testing
            
            IMPORTANT: Your response must be structured as follows:
            
            THINKING: [your analysis of the current state and what action to take]
            
            ACTION: [action name]
            {
                "selector_type": "[id/class/name/xpath/css/tag/link_text]",
                "selector_value": "[value]",
                ... other parameters as needed
            }
            
            Current website state:
            """ + json.dumps(state["website_data"]) + """
            
            Current test step: """ + str(state["current_step"]) + """
            
            Remember to take screenshots at key points to document the testing process.
            """
            
            # Get response from LLM without function calling
            response = self.llm.invoke(
                messages + [HumanMessage(content=instruction_content)]
            )
            
            # Add the AI response to the messages
            messages.append(response)
            
            # Return updated state
            return {"messages": messages, **state}
        
        def action_executor(state: AgentState) -> AgentState:
            """Executes the action specified by the agent."""
            # Get the most recent AI message
            last_message = state["messages"][-1]
            content = last_message.content
            
            # Try to extract the action using regular expressions
            import re
            
            # Pattern to match the ACTION: and JSON block
            action_pattern = r'ACTION:\s*(\w+)\s*(\{.*?\})'
            match = re.search(action_pattern, content, re.DOTALL)
            
            if not match:
                return {
                    **state,
                    "website_data": {
                        **state["website_data"],
                        "last_action": "no_action",
                        "result": "No action was specified by the agent."
                    }
                }
            
            # Extract action name and arguments
            function_name = match.group(1)
            arguments_str = match.group(2)
            
            try:
                # Parse the arguments JSON
                arguments = json.loads(arguments_str)
                
                # Execute the function
                result = self._execute_function(function_name, arguments)
                
                # Update test results
                test_results = state["test_results"].copy()
                test_results.append({
                    "step": state["current_step"],
                    "action": function_name,
                    "arguments": arguments,
                    "result": result
                })
                
                # Create a simple response message
                response_message = AIMessage(content=f"Executed {function_name} with result: {json.dumps(result)}")
                
                # Update state
                return {
                    "messages": state["messages"] + [response_message],
                    "website_data": {
                        **state["website_data"],
                        "last_action": function_name,
                        "result": result
                    },
                    "test_results": test_results,
                    "current_step": state["current_step"] + 1,
                    "error": result.get("error") if not result.get("success", True) else None,
                    "done": state["done"]
                }
            except Exception as e:
                error_message = f"Error executing action: {str(e)}"
                return {
                    **state,
                    "website_data": {
                        **state["website_data"],
                        "last_action": "error",
                        "result": {"success": False, "error": error_message}
                    },
                    "error": error_message
                }
        
        def should_continue(state: AgentState) -> Literal["continue", "human_intervention", "complete"]:
            """Determines if the agent should continue or if it's done."""
            # Check if there's an error
            if state["error"]:
                return "human_intervention"
                
            # Check if the agent has explicitly marked the task as done
            if state["done"]:
                return "complete"
                
            # Check if we've reached the maximum number of steps (prevent infinite loops)
            if state["current_step"] > 20:  # Adjust this number as needed
                return "complete"
                
            return "continue"
        
        # Define the graph edges
        builder = lg.Graph()
        builder.add_node("agent", agent)
        builder.add_node("action_executor", action_executor)
        
        # Add edges
        builder.add_edge("agent", "action_executor")
        builder.add_conditional_edges(
            "action_executor",
            should_continue,
            {
                "continue": "agent",
                "human_intervention": END,
                "complete": END
            }
        )
        
        # Set the entry point
        builder.set_entry_point("agent")
        
        # Compile the graph
        return builder.compile()
    
    def _execute_function(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the specified function with the driver."""
        try:
            # Create the appropriate tool instance and run it
            tool_class = next((t for t in self.tools if t.__name__ == function_name), None)
            if not tool_class:
                return {"success": False, "error": f"Unknown function: {function_name}"}
            
            # Create the tool instance with arguments
            tool_instance = tool_class(**arguments)
            
            # Run the tool with the driver setup
            return tool_instance.run(self.driver_setup)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run(self, website_url: str, custom_initial_state: Optional[AgentState] = None) -> Dict[str, Any]:
        """Run the agent on the specified website.
        
        Args:
            website_url: The URL of the website to test
            custom_initial_state: Optional custom initial state with predefined test instructions
        """
        try:
            # Use custom initial state if provided, otherwise create default state
            if custom_initial_state:
                initial_state = custom_initial_state
            else:
                initial_state = AgentState(
                    messages=[],
                    website_data={"current_url": website_url},
                    test_results=[],
                    current_step=1,
                    error=None,
                    done=False
                )
            
            # First, navigate to the website
            navigate_result = self.driver_setup.navigate_to(website_url)
            initial_state["website_data"] = {**initial_state["website_data"], **navigate_result}
            if not navigate_result["success"]:
                initial_state["error"] = navigate_result["error"]
                return self._generate_report(initial_state)
            
            # Take an initial screenshot
            screenshot_result = self.driver_setup.take_screenshot("Initial page load")
            initial_state["test_results"].append({
                "step": 0,
                "action": "TakeScreenshot",
                "arguments": {"description": "Initial page load"},
                "result": screenshot_result
            })
            
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Generate and return the report
            return self._generate_report(final_state)
        finally:
            # Always close the browser
            self.driver_setup.close()
    
    def _generate_pdf_report(self, report: Dict[str, Any]):
        """Generate a comprehensive PDF report with screenshots."""
        try:
            # Import reportlab libraries
            from reportlab.lib.pagesizes import letter, landscape
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            
            # Create a PDF document
            pdf_path = "testing_report.pdf"
            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
            
            # Create styles
            styles = getSampleStyleSheet()
            title_style = styles["Heading1"]
            heading2_style = styles["Heading2"]
            normal_style = styles["Normal"]
            
            # Custom styles
            header_style = ParagraphStyle(
                'HeaderStyle',
                parent=styles['Heading2'],
                textColor=colors.darkblue,
                spaceAfter=12
            )
            
            # Initialize story list for PDF elements
            story = []
            
            # Add title
            title = Paragraph(f"Video Converter Testing Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}", title_style)
            story.append(title)
            story.append(Spacer(1, 0.25*inch))
            
            # Add summary section
            story.append(Paragraph("Test Summary", header_style))
            
            # Create summary table
            summary_data = [
                ["Test ID", report["test_id"]],
                ["Timestamp", report["test_timestamp"]],
                ["Total Steps", str(report["total_steps"])],
                ["Status", "SUCCESS" if report["success"] else "FAILED"],
            ]
            
            if not report["success"] and report["error"]:
                summary_data.append(["Error", report["error"]])
            
            summary_table = Table(summary_data, colWidths=[2*inch, 4*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.darkblue),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 0.25*inch))
            
            # Add confidence scores
            if "confidence_scores" in report:
                story.append(Paragraph("Confidence Scores", header_style))
                
                scores_data = [["Metric", "Score (out of 100)"]]
                for metric, score in report["confidence_scores"].items():
                    # Convert snake_case to Title Case
                    formatted_metric = " ".join(word.capitalize() for word in metric.split("_"))
                    scores_data.append([formatted_metric, str(score)])
                
                scores_table = Table(scores_data, colWidths=[3*inch, 3*inch])
                scores_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                
                story.append(scores_table)
                story.append(Spacer(1, 0.25*inch))
            
            # Add analysis section
            if "ai_analysis" in report:
                story.append(Paragraph("AI Analysis", header_style))
                story.append(Paragraph(report["ai_analysis"], normal_style))
                story.append(Spacer(1, 0.15*inch))
            
            # Add recommendations section
            if "recommendations" in report and report["recommendations"]:
                story.append(Paragraph("Recommendations", header_style))
                for i, recommendation in enumerate(report["recommendations"], 1):
                    story.append(Paragraph(f"{i}. {recommendation}", normal_style))
                story.append(Spacer(1, 0.25*inch))
            
            # Add steps with screenshots section
            story.append(Paragraph("Test Steps & Screenshots", header_style))
            
            # Create a mapping of screenshot paths to step numbers
            screenshot_map = {}
            for step in report["steps"]:
                if step["action"] == "TakeScreenshot" and step["result"]["success"]:
                    screenshot_map[step["result"]["screenshot_path"]] = {
                        "step": step["step"],
                        "description": step["result"]["description"]
                    }
            
            # Add each test step
            for step_index, step in enumerate(report["steps"]):
                # Add step header
                step_num = step["step"]
                action = step["action"]
                
                # Skip the initial automatic screenshot (step 0)
                if step_num == 0:
                    continue
                    
                story.append(Paragraph(f"Step {step_num}: {action}", heading2_style))
                
                # Add arguments used
                args_text = ", ".join([f"{k}='{v}'" for k, v in step["arguments"].items()])
                story.append(Paragraph(f"Arguments: {args_text}", normal_style))
                
                # Add result summary
                success = step["result"].get("success", False)
                result_color = colors.green if success else colors.red
                result_text = Paragraph(
                    f"Result: {'Success' if success else 'Failed'}", 
                    ParagraphStyle(
                        'ResultStyle',
                        parent=normal_style,
                        textColor=result_color,
                        fontName='Helvetica-Bold'
                    )
                )
                story.append(result_text)
                
                # Add any error messages
                if not success and "error" in step["result"]:
                    story.append(Paragraph(f"Error: {step['result']['error']}", normal_style))
                
                # Add screenshot if this step has one
                if action == "TakeScreenshot" and step["result"]["success"]:
                    path = step["result"]["screenshot_path"]
                    description = step["result"]["description"]
                    
                    # Add description
                    story.append(Paragraph(f"Screenshot: {description}", normal_style))
                    
                    # Add the image with a reasonable size
                    # Scale down large screenshots to fit the page
                    img = Image(path)
                    img.drawHeight = 4*inch
                    img.drawWidth = 6*inch
                    story.append(img)
                
                story.append(Spacer(1, 0.25*inch))
            
            # Build the PDF document
            doc.build(story)
            
            return {"success": True, "pdf_path": pdf_path}
        except Exception as e:
            import traceback
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def _generate_report(self, final_state: AgentState) -> Dict[str, Any]:
        """Generate a comprehensive test report with AI analysis."""
        # Create a unique test ID
        test_id = f"test_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Generate AI analysis based on test results
        ai_analysis = self._generate_ai_analysis(final_state)
        
        # Create a report with all the test results
        report = {
            "test_id": test_id,
            "test_timestamp": datetime.now().isoformat(),
            "total_steps": final_state["current_step"] - 1,
            "success": final_state["error"] is None,
            "error": final_state["error"],
            "screenshots": [
                result["result"]["screenshot_path"]
                for result in final_state["test_results"]
                if result["action"] == "TakeScreenshot" and result["result"]["success"]
            ],
            "steps": final_state["test_results"],
            "ai_analysis": ai_analysis["analysis"],
            "recommendations": ai_analysis["recommendations"],
            "explanation": ai_analysis["explanation"],
            "confidence_scores": ai_analysis["confidence_scores"]
        }
        
        # Save the report to a JSON file
        with open(self.report_path, "w") as f:
            json.dump(report, f, indent=2)
            
        # Generate PDF report with screenshots
        pdf_result = self._generate_pdf_report(report)
        report["pdf_report"] = pdf_result.get("pdf_path") if pdf_result.get("success") else None
        
        # If there are screenshots, generate a visual report (MatPlotLib)
        if report["screenshots"]:
            self._generate_visual_report(report["screenshots"])
                
        return report

    def _generate_ai_analysis(self, final_state: AgentState) -> Dict[str, Any]:
        """Generate AI analysis of the test results."""
        # Extract useful information for analysis
        steps = final_state["test_results"]
        total_steps = final_state["current_step"] - 1
        error = final_state["error"]
        success = error is None
        
        # Count number of successful and failed actions
        successful_actions = sum(1 for step in steps if step["result"].get("success", False))
        failed_actions = sum(1 for step in steps if not step["result"].get("success", False))
        
        # Check for specific action types
        navigation_count = sum(1 for step in steps if step["action"] == "NavigateTo")
        click_count = sum(1 for step in steps if step["action"] == "ClickElement")
        type_count = sum(1 for step in steps if step["action"] == "TypeText")
        alert_interactions = sum(1 for step in steps if step["action"] in ["HandleAlert", "CheckAlertPresent"])
        
        # Look for patterns in errors
        error_patterns = {}
        for step in steps:
            if not step["result"].get("success", False) and "error" in step["result"]:
                error_msg = step["result"]["error"]
                error_patterns[error_msg] = error_patterns.get(error_msg, 0) + 1
        
        # Generate analysis
        analysis = "The automated test "
        if success:
            analysis += "completed successfully with all required steps. "
        else:
            analysis += f"failed at step {total_steps} with error: {error}. "
        
        analysis += f"The test performed {total_steps} steps with {successful_actions} successful actions and {failed_actions} failed actions. "
        
        if alert_interactions > 0:
            analysis += f"The test encountered and handled {alert_interactions} alert dialogs. "
        
        # Generate detailed recommendations based on test outcomes
        recommendations = []
        
        # If test failed, add recommendations to fix the issue
        if not success:
            recommendations.append(f"Fix the error at step {total_steps}: {error}")
            
            # Add more specific recommendations based on error types
            if "not found" in str(error).lower() or "element not found" in str(error).lower():
                recommendations.append("Check if element selectors are correct and if the page structure has changed")
            elif "timeout" in str(error).lower():
                recommendations.append("Increase timeout duration for slow-loading elements")
            elif "alert" in str(error).lower():
                recommendations.append("Add explicit alert handling steps in the test sequence")
        
        # General recommendations for test improvement
        if navigation_count > 1:
            recommendations.append("Consider optimizing test flow to reduce page navigations")
        
        # If successful but took many steps, suggest optimization
        if success and total_steps > 15:
            recommendations.append("Consider optimizing the test by reducing the number of steps")
        
        # Add explanation for the analysis
        explanation = "This analysis is based on the executed test steps and their outcomes. "
        explanation += f"The test interacted with the website through {click_count} clicks, {type_count} text inputs, and {navigation_count} page navigations. "
        
        if error_patterns:
            explanation += "The most common errors encountered were: "
            for error_msg, count in error_patterns.items():
                explanation += f"'{error_msg}' ({count} occurrences), "
            explanation = explanation.rstrip(", ") + ". "
        
        # Generate confidence scores
        website_functionality_score = 0
        test_coverage_score = 0
        reliability_score = 0
        
        # Calculate website functionality score (how well the website worked)
        if success:
            website_functionality_score = 90  # Base score for successful tests
            if failed_actions > 0:
                website_functionality_score -= min(failed_actions * 10, 40)  # Reduce for failed actions
        else:
            # Failed test gets a lower base score
            website_functionality_score = 50 - min(failed_actions * 5, 40)
        
        # Calculate test coverage score
        expected_steps = 7  # Based on the test plan in main()
        if total_steps >= expected_steps:
            test_coverage_score = 90
        else:
            test_coverage_score = (total_steps / expected_steps) * 90
        
        # Calculate reliability score
        if success and failed_actions == 0:
            reliability_score = 95
        elif success:
            reliability_score = 80 - min(failed_actions * 5, 30)
        else:
            reliability_score = 40 - min(failed_actions * 5, 30)
        
        # Ensure scores are within 0-100 range
        website_functionality_score = max(0, min(100, website_functionality_score))
        test_coverage_score = max(0, min(100, test_coverage_score))
        reliability_score = max(0, min(100, reliability_score))
        
        confidence_scores = {
            "website_functionality": round(website_functionality_score),
            "test_coverage": round(test_coverage_score),
            "test_reliability": round(reliability_score)
        }
        
        return {
            "analysis": analysis,
            "recommendations": recommendations,
            "explanation": explanation,
            "confidence_scores": confidence_scores
        }
    
    def _generate_visual_report(self, screenshot_paths: List[str]):
        """Generate a visual report with all the screenshots."""
        num_screenshots = len(screenshot_paths)
        fig, axes = plt.subplots(
            nrows=num_screenshots, 
            ncols=1, 
            figsize=(10, 5 * num_screenshots)
        )
        
        # If there's only one screenshot, axes is not a list
        if num_screenshots == 1:
            axes = [axes]
            
        for i, path in enumerate(screenshot_paths):
            img = Image.open(path)
            axes[i].imshow(img)
            axes[i].set_title(f"Step {i+1}: {os.path.basename(path)}")
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig("visual_report.png")
        plt.close()

# Example usage
def main():
    # Replace with your actual OpenAI API key
    OPENAI_API_KEY ="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" 
    
    # Initialize the agent
    agent = VideoConverterTestingAgent(openai_api_key=OPENAI_API_KEY)
    
    # Define a sample test case with a video URL to convert
    website_url = "https://video-converter.com/"
    video_url = "https://www.sample-videos.com/video321/mp4/360/big_buck_bunny_360p_2mb.mp4"  # Sample video URL
    
    # Modify initial state to include test instructions
    initial_state = AgentState(
        messages=[
            HumanMessage(content=f"""
            I want you to test the video converter website using a URL input.
            
            Here's the testing plan:
            1. Navigate to the video converter website
            2. Click on the URL option (instead of uploading a file)
            3. Enter this video URL: {video_url}
            4. Choose 'mp4' as the output format
            5. Set resolution to '720p' if available
            6. Click on any convert/start button
            7. Wait for and document the result (conversion start, progress, etc.)
            
            Take screenshots at each important step.
            """)
        ],
        website_data={"current_url": website_url, "video_url": video_url},
        test_results=[],
        current_step=1,
        error=None,
        done=False
    )
    
    # Run the agent with the custom test instructions
    report = agent.run(website_url, initial_state)
    
    # Print some statistics from the report
    print(f"Testing completed with {report['total_steps']} steps")
    print(f"Success: {report['success']}")
    if not report['success']:
        print(f"Error: {report['error']}")
    print(f"Screenshots taken: {len(report['screenshots'])}")
    print(f"Full report saved to: {agent.report_path}")
    if report['screenshots']:
        print(f"Visual report saved to: visual_report.png")

    if report.get('pdf_report'):
        print(f"PDF report with screenshots saved to: {report['pdf_report']}")
    if report['screenshots']:
        print(f"Visual report saved to: visual_report.png")

if __name__ == "__main__":
    main()
