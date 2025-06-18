# MM AI Charting 📈📊🍕

An AI-powered financial charting application that helps users search and generate interactive charts from MacroMicro (MM) financial data using Gemini AI.

## Features

- 🔍 Intelligent chart search using natural language queries
- 🎨 Dynamic chart generation with Highcharts
- 💰 Real-time token usage and cost tracking
- 🌐 Interactive web interface with Streamlit
- 📊 Access to comprehensive MacroMicro financial datasets

## How to run it on your own machine

1. Clone the repository
   ```bash
   git clone https://github.com/x1001000/mm-ai-charting.git
   cd mm-ai-charting
   ```

2. Install the requirements
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your secrets
   Create a `.streamlit/secrets.toml` file with:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key"
   CHART_DATA_API = "your_chart_data_api_endpoint"
   ```

4. Run the app
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage

1. Enter a financial chart query in Chinese or English (e.g., "小台散戶", "柯博文指數", "美債殖利率vs基準利率")
2. Click the "✨生成圖表" button (automatically enabled when text is entered)
3. Wait for AI to search, load, and generate the interactive chart
4. View token usage and costs in the sidebar

## Technologies Used

- **Streamlit** - Web application framework
- **Google Gemini AI** - Chart search and code generation
- **Highcharts** - Interactive chart rendering
- **Gradio Client** - API integration for chart data
- **MacroMicro** - Financial data source
