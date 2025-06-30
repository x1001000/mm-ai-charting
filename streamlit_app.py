import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from gradio_client import Client
if 'gradio_client' not in st.session_state:
    st.session_state.gradio_client = Client(st.secrets['CHART_DATA_API'])
    st.session_state.charts = st.session_state.gradio_client.predict(api_name="/get_all_charts")

from google import genai
from google.genai import types
client = genai.Client(api_key=st.secrets['GEMINI_API_KEY'])
thinking_model = 'gemini-2.5-flash' # nonthinking model is stupid af
price = {
    'gemini-2.5-flash': {'input': 0.3, 'output': 2.5, 'thinking': 2.5, 'caching': 0.075},
    'gemini-2.5-pro-preview-06-05': {'input': 1.25, 'output': 10, 'thinking': 10, 'caching': 0.31},
}

def extract_tokens(usage_metadata):
    d = usage_metadata.model_dump()
    result = {}
    for key in d:
        if 'token_count' in key:
            result[key.replace('token_count', 'tokens')] = d[key] if d[key] else 0
    return result

def calculate_cost(tokens, model_name):
    return round((tokens['prompt_tokens'] * price[model_name]['input'] + tokens['candidates_tokens'] * price[model_name]['output'] + tokens['thoughts_tokens'] * price[model_name]['thinking'])/1e6, 3)

def generate_content(model, user_prompt, system_prompt, response_type, response_schema, tools):
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type=response_type,
            response_schema=response_schema,
            tools=tools,
        )
    )
    return response

st.title("MM AI Charting ![](https://cdn.macromicro.me/assets/img/favicons/favicon-32.png)✨📈")

# Initialize session state for current request tokens
if "current_request_tokens" not in st.session_state:
    st.session_state.current_request_tokens = {'prompt_tokens': 0, 'candidates_tokens': 0, 'cached_content_tokens': 0, 'thoughts_tokens': 0, 'tool_use_prompt_tokens': 0, 'total_tokens': 0}

# initialize
if 'contents' not in st.session_state:
    st.session_state.contents = []
    st.session_state.df = None

for content in st.session_state.contents:
    with st.chat_message(content.role):
        st.markdown(content.parts[0].text)

# Chat input
prompt = st.chat_input("您上傳的時間序列，想和什麼MM總經數據一起呈現？試試：台幣", accept_file=True, file_type=["csv"])
if prompt and prompt.text:
    user_prompt = prompt.text
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]))
if prompt and prompt.files:
    file = prompt.files[0]
    df = pd.read_csv(file)
    st.session_state.df = df

if st.session_state.df is not None:
    st.session_state.df = st.data_editor(
        st.session_state.df, 
        num_rows="dynamic",
        key="data_editor"
    )
    
    if st.button("✅ 確認數據並生成圖表", type="primary"):
        if "data_editor" in st.session_state and st.session_state.data_editor is not None:
            # Save edited data to CSV
            st.session_state.df.to_csv("uploaded_series.csv", index=False)
            st.success("📁 已保存編輯後的數據到 uploaded_series.csv")
            # Loading UI
            with st.spinner("🔍 正在搜尋相關圖表..."):
                model = thinking_model
                user_prompt = st.session_state.contents[-1].parts[0].text
                system_prompt = 'Find the most relevant chart id for the user query. Output the id.\n\n' + st.session_state.charts
                response_type = 'application/json'
                response_schema = types.Schema(type = genai.types.Type.STRING)
                tools = None
                response = generate_content(model, user_prompt, system_prompt, response_type, response_schema, tools)
                chart_id = response.parsed
                print(chart_id)
                if not chart_id.isdigit(): # in case of null
                    st.error('Houston, we have a problem. Please try again!', icon="🚨")
                    st.stop()
        
                # Extract tokens from first API call
                tokens = extract_tokens(response.usage_metadata)
                for key in tokens:
                    st.session_state.current_request_tokens[key] += tokens[key]

            with st.spinner("⚙️ 正在載入圖表配置..."):
                chart_info_output, sample_series_output, series_api_output= st.session_state.gradio_client.predict(
                        chart_id=chart_id,
                        api_name="/get_one_chart"
                )
                import json
                chart_info = json.loads(chart_info_output)
                series_sample = json.loads(sample_series_output)
                del chart_info['description_en'] # avoid single quote in description_en causing error in text generation
                series_configs = chart_info['chart_config']['seriesConfigs']
                retrieval = {
                    # "chart_info": chart_info,
                    "series_configs": series_configs,
                    "series_sample": series_sample,
                    "series_api": series_api_output,
                    # "MM Chart reference": f"[{chart_info['name_tc']}](https://www.macromicro.me/charts/{chart_id}/{chart_info['slug']})"
                }
                from pprint import pprint
                pprint(retrieval)

            with st.spinner("✍️ 正在撰寫 Plotly 程式..."):
                # system_prompt = 'Retrieval data is as below. Customized by user input, generate Highchart HTML/JS/CSS source code which calls the series API to get the full series replacing sample series and has a button link to MM Chart reference. Write the code in multilines without code comments. Output only the Highchart HTML/JS/CSS source code.\n\n' +\
                #     json.dumps(retrieval, ensure_ascii=False)
                system_prompt = """You are tasked with coding a Python module that plots a Plotly chart. Your module should:

1. Use the retrieval data provided below for chart context and configuration
2. Fetch full series data by requests.get the series_api and replace the series_sample in the retrieval data
3. pandas.read_csv('uploaded_series.csv') into a DataFrame variable named `user_series_df` and add into the chart together with the full series
4. Code this Plotly chart that combines points mentioned above and customizations based on user input text prompt
5. Your main() function should use streamlit.plotly_chart(fig) to display the chart.

Retrieval data:
""" + json.dumps(retrieval, ensure_ascii=False)
                model = thinking_model
                response_type = 'application/json'
                response_schema = types.Schema(type = genai.types.Type.STRING)
                tools = None
                response = generate_content(model, user_prompt, system_prompt, response_type, response_schema, tools)
                # highchart_code = response.parsed
                plotly_module = response.parsed
                with open('plotly_module.py', 'w') as f:
                    # https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart
                    f.write(plotly_module.replace('\\n', '\n')) # replace escaped newlines with actual newlines
                
                # Extract tokens from second API call
                tokens = extract_tokens(response.usage_metadata)
                for key in tokens:
                    st.session_state.current_request_tokens[key] += tokens[key]

            st.success("✅ 圖表生成完成！（若出現 Error 或有 🐛 就再按一次吧）")
            # components.html(highchart_code, height=700)
            try:
                import plotly_module
                plotly_module.main()
                st.markdown('### Reference：')
                st.link_button(chart_info['name_tc'], url=f"https://www.macromicro.me/charts/{chart_id}/{chart_info['slug']}", icon="📊")
            except Exception as e:
                st.error(f"Error: {e}")

        # Display token count and cost in sidebar
        with st.sidebar:
            st.header("✨")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Input Tokens", st.session_state.current_request_tokens['prompt_tokens'])
                st.metric("Cached Tokens", st.session_state.current_request_tokens['cached_content_tokens'])
                st.metric("Tool Use Tokens", st.session_state.current_request_tokens['tool_use_prompt_tokens'])
            with col2:
                st.metric("Output Tokens", st.session_state.current_request_tokens['candidates_tokens'])
                st.metric("Thinking Tokens", st.session_state.current_request_tokens['thoughts_tokens'])
                st.metric("Total Tokens", st.session_state.current_request_tokens['total_tokens'])
            '---'
            col1, col2 = st.columns(2)
            with col1:
                pass
            with col2:
                st.header("💰")
                request_cost = calculate_cost(st.session_state.current_request_tokens, thinking_model)
                st.metric("Cost", f"${request_cost}")

            # with st.chat_message("user"):
            #     st.markdown(prompt)
            
            # # Generate response with Gemini
            # with st.chat_message("assistant"):
            #     try:
            #         response = client.models.generate_content(prompt)
            #         response_text = response.text
            #         st.markdown(response_text)
                    
            #         # Add assistant response to chat history
            #         st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
            #     except Exception as e:
            #         error_msg = f"Error: {str(e)}"
            #         st.error(error_msg)
            #         st.session_state.messages.append({"role": "assistant", "content": error_msg})