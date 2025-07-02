import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json

from gradio_client import Client
if 'gradio_client' not in st.session_state:
    st.session_state.gradio_client = Client(st.secrets['CHART_DATA_API'])
    st.session_state.charts = st.session_state.gradio_client.predict(api_name="/get_all_charts")
    st.session_state.charts_list = json.loads(st.session_state.charts)

from google import genai
from google.genai import types
client = genai.Client(api_key=st.secrets['GEMINI_API_KEY'])
finder_model = 'gemini-2.5-flash'
writer_model = 'gemini-2.5-pro'
price = {
    'gemini-2.5-flash': {'input': 0.3, 'output': 2.5, 'thinking': 2.5, 'caching': 0.075},
    'gemini-2.5-pro': {'input': 1.25, 'output': 10, 'thinking': 10, 'caching': 0.31},
}

import system_prompts
import importlib
import sys
import random
chart_name = random.choice(st.session_state.charts_list)['name_tc']

def extract_tokens(usage_metadata):
    d = usage_metadata.model_dump()
    result = {}
    for key in d:
        if 'token_count' in key:
            result[key.replace('token_count', 'tokens')] = d[key] if d[key] else 0
    return result

def calculate_cost(tokens, model_name):
    return round((tokens.get('prompt_tokens', 0) * price[model_name]['input'] + tokens.get('candidates_tokens', 0) * price[model_name]['output'] + tokens.get('thoughts_tokens', 0) * price[model_name]['thinking'])/1e6, 3)

def generate_chart(user_query, has_csv_data=False):
    st.session_state.current_request_cost = 0
    chart_info = None
    chart_id = None
    retrieval = '- series_api is not available.'
    if user_query:
        # Find relevant chart
        with st.spinner("🔍 正在檢索相關MM圖表..."):
            response = client.models.generate_content(
                model=finder_model,
                contents=user_query,
                config=types.GenerateContentConfig(
                    system_instruction='Find the most relevant chart id for the user query. Output the id.\n\n' + st.session_state.charts,
                    response_mime_type='application/json',
                    response_schema=types.Schema(type = genai.types.Type.STRING),
                    tools=None,
                    temperature=0.2,
                )
            )
            chart_id = response.parsed
            print(chart_id)

            # Extract tokens from first API call
            tokens = extract_tokens(response.usage_metadata)
            st.session_state.current_request_cost += calculate_cost(tokens, finder_model)
            for key in tokens:
                st.session_state.current_request_tokens[key] += tokens[key]

        if chart_id and chart_id.isdigit():
            # Load chart configuration
            with st.spinner("⚙️ 正在處理MM圖表序列資料..."):
                chart_info_output, series_sample_output, series_api_output = st.session_state.gradio_client.predict(
                        chart_id=chart_id,
                        api_name="/get_one_chart"
                )
                chart_info = json.loads(chart_info_output)
                series_sample = json.loads(series_sample_output)
                del chart_info['description_en']
                series_names = [series_config['stats'][0]['name_en'] for series_config in chart_info['chart_config']['seriesConfigs']]
                retrieval = {
                    "series_names": series_names,
                    "series_api": series_api_output,
                }
                from pprint import pprint
                pprint(retrieval)

    # TT, TF, FT : 3 modes
    if user_query and has_csv_data:
        system_prompt = system_prompts.api_csv + f"\n\n{json.dumps(retrieval, ensure_ascii=False)}"
    elif user_query and not has_csv_data:
        system_prompt = system_prompts.api_only + f"\n\n{json.dumps(retrieval, ensure_ascii=False)}"
    else:
        system_prompt = system_prompts.csv_only

    # Generate chart code
    with st.spinner("✍️ 正在撰寫 Plotly 程式..."):
        response = client.models.generate_content(
            model=writer_model,
            contents=user_query,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type='application/json',
                response_schema=genai.types.Schema(
                    type = genai.types.Type.OBJECT,
                    properties = {
                        "plotly_module": genai.types.Schema(
                            type = genai.types.Type.STRING,
                        ),
                    },
                ),
                tools=None,
                temperature=0.2,
            )
        )
        plotly_module = response.parsed['plotly_module']
        plotly_module = plotly_module.replace('\\n', '\n')
        plotly_module = plotly_module.replace('```python', '').replace('```', '')
        with open('plotly_module.py', 'w') as f:
            f.write(plotly_module)
        
        # Extract tokens from second API call
        tokens = extract_tokens(response.usage_metadata)
        st.session_state.current_request_cost += calculate_cost(tokens, writer_model)
        for key in tokens:
            st.session_state.current_request_tokens[key] += tokens[key]

    st.success("✅ 圖表繪製完成！（若出現 Error 或有 🐛 就再按一次吧）")
    st.session_state.chart_ready = True
    st.session_state.chart_info = chart_info
    st.session_state.chart_id = chart_id

'![](https://cdn.macromicro.me/assets/img/favicons/favicon-32.png)✨📈👩🏻‍🔬'
st.title("MM AI Charting Lab")

# Initialize session state for current request tokens
if "current_request_tokens" not in st.session_state:
    st.session_state.current_request_tokens = {'prompt_tokens': 0, 'candidates_tokens': 0, 'cached_content_tokens': 0, 'thoughts_tokens': 0, 'tool_use_prompt_tokens': 0, 'total_tokens': 0}
if "current_request_cost" not in st.session_state:
    st.session_state.current_request_cost = 0

# initialize
if 'contents' not in st.session_state:
    st.session_state.contents = []
    st.session_state.df = None

for content in st.session_state.contents:
    with st.chat_message(content.role):
        st.markdown(content.parts[0].text)

# Chat input
prompt = st.chat_input("您上傳的CSV（第一欄為日期、第一列為序列名稱），想和什麼MM總經圖表數據一起呈現？試試：美國非農就業", accept_file=True, file_type=["csv"])
if prompt and prompt.text:
    user_prompt = prompt.text
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]))
if prompt and prompt.files:
    file = prompt.files[0]
    df = pd.read_csv(file)
    st.session_state.df = df

# use API data
if st.session_state.df is None and st.session_state.contents:
    if st.button("不上傳數據，直接生成圖表 🚀", type="primary"):
        generate_chart(st.session_state.contents[-1].parts[0].text, has_csv_data=False)

# use CSV data
if st.session_state.df is not None: # The truth value of a DataFrame is ambiguous
    edited_df = st.data_editor(
        st.session_state.df, 
        num_rows="dynamic",
        key="data_editor"
    )
    if st.button("數據編輯完成，開始生成圖表 🚀", type="primary"):
        # Save edited data to CSV for generated module to read
        edited_df.to_csv("uploaded_series.csv", index=False)
        
        # use CSV+API data
        if st.session_state.contents:
            generate_chart(st.session_state.contents[-1].parts[0].text, has_csv_data=True)
        
        # use CSV data only
        else:
            generate_chart('', has_csv_data=True)

# Display token count and cost in sidebar
with st.sidebar:
            st.header("🧠")
            st.metric('MM Chart finder model', finder_model)
            st.metric('Plotly Code writer model', writer_model)
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
                st.metric("Cost", f"${st.session_state.current_request_cost:.3f}")

# Display chart if ready
if hasattr(st.session_state, 'chart_ready') and st.session_state.chart_ready:
    try:
        import plotly_module
        if 'plotly_module' in sys.modules:
            importlib.reload(sys.modules['plotly_module'])
        else:
            import plotly_module
        plotly_module.main()
        if st.session_state.chart_info:
            '相關MM圖表'
            st.link_button(st.session_state.chart_info['name_tc'], 
                         url=f"https://www.macromicro.me/charts/{st.session_state.chart_id}/{st.session_state.chart_info['slug']}", 
                         icon="📊")
    except Exception as e:
        st.error(f"Error: {e}")
