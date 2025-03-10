# TesterAI

# Video Converter Testing Agent

An automated testing framework for web-based video converter applications built with Python, Selenium, and LangGraph.

## Overview

This project provides an AI-powered agent that automatically tests video converter websites. The agent navigates through the website, interacts with elements, handles JavaScript alerts, and documents the process with screenshots. It generates comprehensive reports in both JSON and PDF formats.

## Key Features

- **AI-Powered Testing**: Uses GPT-4o to intelligently navigate and test website functionality
- **Automated Test Execution**: Performs clicks, text input, navigation, and more without human intervention
- **Alert Handling**: Detects and manages JavaScript alert boxes that appear during testing
- **Comprehensive Reporting**: Generates detailed PDF reports with screenshots, analysis, and recommendations
- **Visual Documentation**: Takes screenshots at key testing steps for visual verification
- **Error Recovery**: Handles common web testing errors with appropriate fallbacks
- **Confidence Scoring**: Provides quantitative assessment of website functionality, test coverage, and reliability

## Requirements

- Python 3.8+
- OpenAI API key
- Chrome WebDriver

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/video-converter-testing-agent.git
cd video-converter-testing-agent
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install Chrome WebDriver:
   - Download Chrome WebDriver from [https://sites.google.com/chromium.org/driver/](https://sites.google.com/chromium.org/driver/)
   - Make sure it matches your Chrome version
   - Add it to your system PATH or specify its location in the code

## Usage

1. Set your OpenAI API key in the `main()` function:

```python
OPENAI_API_KEY = "your-openai-api-key-here"
```

2. Modify the testing plan in the `main()` function to suit your needs:

```python
website_url = "https://video-converter.com/"  # Replace with the website URL
video_url = "https://example.com/sample-video.mp4"  # Replace with a sample video URL
```

3. Run the script:

```bash
python video_converter_testing_agent.py
```

4. Check the generated reports:
   - `testing_report.json`: Contains all test steps and results
   - `testing_report.pdf`: Comprehensive PDF report with screenshots and analysis
   - `visual_report.png`: Visual summary of screenshots

## Custom Test Plans

You can create custom test plans by modifying the `initial_state` in the `main()` function:

```python
initial_state = AgentState(
    messages=[
        HumanMessage(content="""
        I want you to test the following workflow:
        1. Step 1 description
        2. Step 2 description
        ...
        """)
    ],
    website_data={"current_url": website_url, "custom_data": "your_data"},
    test_results=[],
    current_step=1,
    error=None,
    done=False
)
```

## Architecture

The agent uses the following architecture:

1. **DriverSetup Class**: Handles Selenium WebDriver interactions with the website
2. **Tool Classes**: Define specific web actions (click, type, screenshot, etc.)
3. **LangGraph Workflow**: Coordinates the AI's decision-making process
4. **Reporting System**: Generates comprehensive test reports

## Available Tools

The agent has access to the following tools for website interaction:

- **FindElement**: Locate elements on the page
- **ClickElement**: Click on buttons or links
- **TypeText**: Enter text into input fields
- **TakeScreenshot**: Capture the current state of the page
- **NavigateTo**: Go to a specific URL
- **WaitFor**: Wait for elements to appear
- **GetPageSource**: Retrieve the page's HTML
- **HandleAlert**: Interact with JavaScript alert boxes
- **CheckAlertPresent**: Check for the presence of alerts

## Report Details

The generated reports include:

- **Test Summary**: Overview of test execution and results
- **Step-by-Step Execution**: Detailed record of each step with screenshots
- **AI Analysis**: Automated analysis of test results
- **Recommendations**: Suggestions for improving website functionality or test coverage
- **Confidence Scores**: Quantitative metrics for website functionality, test coverage, and reliability

## Examples

Here's a sample of how to run a test for a video converter site:

```python
from video_converter_testing_agent import VideoConverterTestingAgent, AgentState
from langchain_core.messages import HumanMessage

agent = VideoConverterTestingAgent(openai_api_key="your-api-key")

# Define test parameters
website_url = "https://video-converter.com/"
video_url = "https://example.com/sample.mp4"

# Create test instructions
initial_state = AgentState(
    messages=[
        HumanMessage(content=f"""
        Test the video converter with the following steps:
        1. Navigate to the site
        2. Click on URL input option
        3. Enter video URL: {video_url}
        4. Select MP4 format
        5. Start conversion
        6. Verify conversion begins
        """)
    ],
    website_data={"current_url": website_url, "video_url": video_url},
    test_results=[],
    current_step=1,
    error=None,
    done=False
)

# Run the test
report = agent.run(website_url, initial_state)

# Print results
print(f"Test successful: {report['success']}")
```

## Troubleshooting

If you encounter issues:

1. **Chrome Driver Errors**: Ensure your Chrome WebDriver version matches your Chrome browser version
2. **API Key Issues**: Verify your OpenAI API key is correct and has sufficient credits
3. **Element Not Found Errors**: Website structure may have changed; adjust selectors or use more robust XPath
4. **Alert Handling Problems**: For sites with complex alert patterns, consider custom alert handling functions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
