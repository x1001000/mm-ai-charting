import streamlit as st
st.set_page_config(layout="wide")
import streamlit_highcharts as hct
import pandas as pd
import numpy as np
import json
import random
import requests
from pprint import pprint

@st.cache_data  # 👈 Add the caching decorator
def load_data(url):
    df = pd.read_csv(url)
    return df

if 'random_chart' not in st.session_state:
    df = load_data(st.secrets['CHARTS_DATA_CSV'])
    st.session_state.random_chart = random.choice(df['name_tc'])
    st.session_state.charts = df.iloc[:,:2].to_json(orient='records', force_ascii=False)
    st.session_state.all_series = []
    st.session_state.all_series_sample = []

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

def update_tokens_cost(response):
    tokens = extract_tokens(response.usage_metadata)
    for key in tokens:
        st.session_state.current_request_tokens[key] += tokens[key]
    st.session_state.current_request_cost += calculate_cost(tokens, response.model_version)

def convert_to_float(all_series):
    converted = []
    for series in all_series:
        converted_series = {
            'name': series['name'],
            'data': [[date_str, float(value_str)] for date_str, value_str in series['data']]
        }
        converted.append(converted_series)
    return converted

def generate_highcharts_options(chart_of_interest, how_to_chart):
    # 2nd API call
    if chart_of_interest:
        response = client.models.generate_content(
            model=model,
            contents=chart_of_interest,
            config=types.GenerateContentConfig(
                system_instruction='Find the most relevant MacroMicro chart id if the user requests for some statistic. By relevant I mean the statistic type, not the chart type. Output the id or output nothing if the user does not request for some statistic.\n\n' + st.session_state.charts,
                response_mime_type='application/json',
                response_schema=types.Schema(type = genai.types.Type.STRING),
                tools=None,
                temperature=0.2,
            )
        )
        chart_id = response.parsed
        print('response.parsed chart_id:', chart_id)
        update_tokens_cost(response)

        if chart_id and chart_id.isdigit() and st.session_state.chart_id != chart_id:
            # Retrieve chart data from API
            r = requests.get(f"{st.secrets['CHARTS_DATA_API']}/{chart_id}")
            d = r.json()
            chart_info = d['data'][f'c:{chart_id}']['info']
            st.session_state.chart_id = chart_id
            st.session_state.chart_info = chart_info
            all_series_data = d['data'][f'c:{chart_id}']['series']
            all_series_names = [series_config['stats'][0]['name_tc'] for series_config in chart_info['chart_config']['seriesConfigs']]
            all_series = st.session_state.all_series
            all_series_sample = st.session_state.all_series_sample
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

    if not st.session_state.all_series_sample:
        return

    # 3rd API call
    system_prompt = f"At the user's request, generate Highcharts options on retrieval data:\n{json.dumps(st.session_state.all_series_sample, ensure_ascii=False)}"
    print('\n# 3rd API call system_prompt:\n', system_prompt)

    with st.status("生成 Highcharts 程式..."):
        max_retries = 3
        for attempt in range(max_retries):
            response = client.models.generate_content(
                model=model,
                contents=how_to_chart if how_to_chart else 'Help me generate Highcharts options for the data provided by the best practices.',
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
                    thinking_config=types.ThinkingConfig(thinking_budget=4000),
                )
            )
            try:
                highcharts_options = json.loads(response.parsed['highcharts_options'])
                break
            except (json.JSONDecodeError, KeyError) as e:
                if attempt == max_retries - 1:
                    st.error(f"Error parsing Highcharts options after {max_retries} attempts: {e}")
                    return None
                st.warning(f"Attempt {attempt + 1} failed, retrying... Error: {e}")
                update_tokens_cost(response)
        print('\n# highcharts_options')
        pprint(highcharts_options)
        del highcharts_options['series']
        st.code(json.dumps(highcharts_options, indent=4, ensure_ascii=False), language='json')
    update_tokens_cost(response)

    highcharts_options['series'] = convert_to_float(st.session_state.all_series)
    return highcharts_options

# Function declarations for tool use
function_declarations = [
    types.FunctionDeclaration(
        name="generate_highcharts_options",
        description="Generate Highcharts options at the user's request about chart of interest and/or how to chart, on retrieval data.",
        parameters=types.Schema(
            type=genai.types.Type.OBJECT,
            properties={
                "chart_of_interest": genai.types.Schema(type=genai.types.Type.STRING, description="The user's request about chart of interest."),
                "how_to_chart": genai.types.Schema(type=genai.types.Type.STRING, description="The user's request about how to chart."),
            },
            # required=["chart_of_interest", "how_to_chart"],
        )
    ),
]

# Initialize session state for current request tokens
if "current_request_tokens" not in st.session_state:
    st.session_state.current_request_tokens = {'prompt_tokens': 0, 'candidates_tokens': 0, 'cached_content_tokens': 0, 'thoughts_tokens': 0, 'tool_use_prompt_tokens': 0, 'total_tokens': 0}
if "current_request_cost" not in st.session_state:
    st.session_state.current_request_cost = 0

# initialize
if 'contents' not in st.session_state:
    st.session_state.contents = []
    st.session_state.table_name = None
    st.session_state.table_uploaded = None
    st.session_state.chart_id = None
    st.session_state.chart_info = None
    st.session_state.chart_generated = None

st.title("MM AI Charting")

def display_table(filename):
    edited_df = st.data_editor(
        st.session_state.table_uploaded, 
        num_rows="dynamic",
        key=filename
    )
    columns = edited_df.columns.tolist()
    all_series_names = columns[1:]
    columns[0] = 'date'
    edited_df.columns = columns

    all_series = []
    all_series_sample = []
    for name in all_series_names:                                   # difference
        data = edited_df[['date', name]].dropna().values.tolist()   # difference
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
    st.session_state.all_series = all_series
    st.session_state.all_series_sample = all_series_sample

# display table if uploaded
if st.session_state.table_uploaded is not None: # The truth value of a DataFrame is ambiguous
    display_table(st.session_state.table_name)

# display chart if generated
if st.session_state.chart_generated:
    hct.streamlit_highcharts(st.session_state.chart_generated, 600)

# display chat messages
# for content in st.session_state.contents[-2:]:
#     with st.chat_message(content.role, avatar=None if content.role == "user" else 'favicon-32.png'):
#         st.markdown(content.parts[0].text)

# Chat input
placeholder = f"上傳您的CSV（第一欄日期、第一列序列名稱），和我們的總經圖表數據一起繪製！來都來了，試試 👉 {st.session_state.random_chart}"
if prompt := st.chat_input(placeholder, accept_file=True, file_type=["csv"]):
    if prompt.files:
        file = prompt.files[0]
        df = pd.read_csv(file)
        st.session_state.table_uploaded = df
        if st.session_state.table_name != file.name:
            st.session_state.table_name = file.name
            display_table(st.session_state.table_name)
            st.session_state.contents.append(types.Content(role="model", parts=[types.Part.from_text(text='The table for AI charting is uploaded successfully!')]))
    if prompt.text:
        with st.chat_message("user"):
            st.markdown(prompt.text)
        st.session_state.contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt.text)]))
        
        # 1st API call
        system_prompt = "You are MM AI Charting Agent. Your one and only mission is to generate Highcharts at the user's request. If the user prompt is not related to charting, you should claim your mission. 回覆中文時使用繁體中文。"
        print('\n# 1st API call system_prompt:\n', system_prompt)
        pprint(st.session_state.contents[-2:])
        response = client.models.generate_content(
            model=model,
            contents=st.session_state.contents[-2:],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type='text/plain',
                response_schema=None,
                tools=[types.Tool(function_declarations=function_declarations)],
                temperature=0.2,
            )
        )
        if tool_call := response.candidates[0].content.parts[0].function_call:
            if tool_call.name == "generate_highcharts_options":
                print('\n# tool_call.args')
                pprint(tool_call.args)
                chart_of_interest = tool_call.args.get('chart_of_interest')
                how_to_chart = tool_call.args.get('how_to_chart')
                if highcharts_options := generate_highcharts_options(chart_of_interest, how_to_chart):
                    st.session_state.chart_generated = highcharts_options
                    hct.streamlit_highcharts(st.session_state.chart_generated, 600)
                else:
                    with st.chat_message("ai", avatar='favicon-32.png'):
                        st.markdown(placeholder)
        else:
            with st.chat_message("ai", avatar='favicon-32.png'):
                st.markdown(response.candidates[0].content.parts[0].text)
            # st.session_state.contents.append(types.Content(role="model", parts=[types.Part.from_text(text=response.candidates[0].content.parts[0].text)]))
            # st.session_state.contents.append(response.candidates[0].content)
        update_tokens_cost(response)

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
    if st.session_state.chart_info:
        st.metric('圖表數據來源', '')
        st.link_button(st.session_state.chart_info['name_tc'], 
            url=f"https://www.macromicro.me/charts/{st.session_state.chart_id}/{st.session_state.chart_info['slug']}", 
            icon="📈")