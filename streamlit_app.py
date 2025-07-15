import streamlit as st
import streamlit_highcharts as hct
import pandas as pd
import numpy as np
import json
import random
import requests

@st.cache_data  # 👈 Add the caching decorator
def load_data(url):
    df = pd.read_csv(url)
    return df

if 'random_chart' not in st.session_state:
    df = load_data(st.secrets['CHARTS_DATA_CSV'])
    st.session_state.charts = df.iloc[:,:2].to_json(orient='records', force_ascii=False)
    st.session_state.random_chart = random.choice(df['name_tc'])

from google import genai
from google.genai import types
client = genai.Client(api_key=st.secrets['GEMINI_API_KEY'])

price = {
    'gemini-2.5-flash': {'input': 0.3, 'output': 2.5, 'thinking': 2.5, 'caching': 0.075},
    'gemini-2.5-pro': {'input': 1.25, 'output': 10, 'thinking': 10, 'caching': 0.31},
}
model = 'gemini-2.5-flash'

def extract_tokens(usage_metadata):
    d = usage_metadata.model_dump()
    result = {}
    for key in d:
        if 'token_count' in key:
            result[key.replace('token_count', 'tokens')] = d[key] if d[key] else 0
    return result

def calculate_cost(tokens, model_name):
    return round((tokens.get('prompt_tokens', 0) * price[model_name]['input'] + tokens.get('candidates_tokens', 0) * price[model_name]['output'] + tokens.get('thoughts_tokens', 0) * price[model_name]['thinking'])/1e6, 3)

def convert_data(all_series):
    converted = []
    for series in all_series:
        converted_series = {
            'name': series['name'],
            'data': [[date_str, float(value_str)] for date_str, value_str in series['data']]
        }
        converted.append(converted_series)
    return converted

def generate_chart(user_query, has_csv_data=False):
    st.session_state.current_request_cost = 0
    chart_info = None
    chart_id = None
    if user_query:
        # Find relevant chart
        with st.spinner("🔍 正在檢索相關MM圖表..."):
            response = client.models.generate_content(
                model=model,
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
            st.session_state.current_request_cost += calculate_cost(tokens, model)
            for key in tokens:
                st.session_state.current_request_tokens[key] += tokens[key]

        if chart_id and chart_id.isdigit():
            # Load chart configuration
            with st.spinner("⚙️ 正在處理MM圖表序列資料..."):
                r = requests.get(f"{st.secrets['CHARTS_DATA_API']}/{chart_id}")
                d = r.json()
                chart_info = d['data'][f'c:{chart_id}']['info']
                all_series_data = d['data'][f'c:{chart_id}']['series']
                all_series_names = [series_config['stats'][0]['name_tc'] for series_config in chart_info['chart_config']['seriesConfigs']]
                all_series = []
                all_series_sample = []
                for name, data in zip(all_series_names, all_series_data):
                    series = dict()
                    series['name'] = name
                    series['data'] = data
                    all_series.append(series)
                    series = dict()
                    series['name'] = name
                    if len(data) > 10:
                        indices = np.linspace(0, len(data)-1, 10, dtype=int)
                        series['data'] = [data[i] for i in indices]
                    else:
                        series['data'] = data
                    all_series_sample.append(series)

    # TT, TF, FT : 3 modes
    if user_query and has_csv_data:
        system_prompt# = system_prompts.api_csv + f"\n\n{json.dumps(retrieval, ensure_ascii=False)}"
    elif user_query and not has_csv_data:
        system_prompt = f'Generate Highcharts options based on the user query and the relevant MM chart series sample:\n{json.dumps(all_series_sample, ensure_ascii=False)}'
    else:
        system_prompt# = system_prompts.csv_only

    # Generate Highcharts options
    with st.status("生成 Highcharts 程式..."):
        response = client.models.generate_content(
            model=model,
            contents=user_query,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type='application/json',
                response_schema=genai.types.Schema(
                    type = genai.types.Type.OBJECT,
                    properties = {
                        "highcharts_options": genai.types.Schema(
                            type = genai.types.Type.STRING,
                        ),
                    },
                ),
                tools=None,
                temperature=0.2,
            )
        )
        highcharts_options = json.loads(response.parsed['highcharts_options'])
        # from pprint import pprint
        # pprint(highcharts_options)
        del highcharts_options['series']
        st.code(json.dumps(highcharts_options, indent=4, ensure_ascii=False), language='json')
        highcharts_options['series'] = convert_data(all_series)
        
        # Extract tokens from second API call
        tokens = extract_tokens(response.usage_metadata)
        st.session_state.current_request_cost += calculate_cost(tokens, model)
        for key in tokens:
            st.session_state.current_request_tokens[key] += tokens[key]

        st.session_state.chart_ready = True
        st.session_state.chart_info = chart_info
        st.session_state.chart_id = chart_id
        st.session_state.options = highcharts_options

st.title("![](https://cdn.macromicro.me/assets/img/favicons/favicon-32.png)✨Charting Lab")

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
    with st.chat_message(content.role, avatar=None if content.role == "user" else '📊'):
        st.markdown(content.parts[0].text)

# Chat input
prompt = st.chat_input(f"您上傳的CSV（第一欄為日期、第一列為序列名稱），想和什麼MM總經圖表數據一起呈現？試試：{st.session_state.random_chart}", accept_file=True, file_type=["csv"])
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
            st.metric('Model', model)
            '---'
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Input Tokens", st.session_state.current_request_tokens['prompt_tokens'])
                st.metric("Output Tokens", st.session_state.current_request_tokens['candidates_tokens'])
            with col2:
                st.metric("Cached Tokens", st.session_state.current_request_tokens['cached_content_tokens'])
                st.metric("Thinking Tokens", st.session_state.current_request_tokens['thoughts_tokens'])
            with col3:
                st.metric("Tool Use Tokens", st.session_state.current_request_tokens['tool_use_prompt_tokens'])
                st.metric("Total Tokens", st.session_state.current_request_tokens['total_tokens'])
            '---'
            st.metric("Cost", f"${st.session_state.current_request_cost:.3f}")
            '---'

# Display chart if ready
if 'chart_ready' in st.session_state and st.session_state.chart_ready:
    try:
        hct.streamlit_highcharts(st.session_state.options, 600)
        if st.session_state.chart_info:
            st.markdown('MacroMicro 相關圖表')
            st.link_button(st.session_state.chart_info['name_tc'], 
                         url=f"https://www.macromicro.me/charts/{st.session_state.chart_id}/{st.session_state.chart_info['slug']}", 
                         icon="📊")
    except Exception as e:
        st.error(f"Error: {e}")