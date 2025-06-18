import streamlit as st
import streamlit.components.v1 as components

from gradio_client import Client
if 'gradio_client' not in st.session_state:
    st.session_state.gradio_client = Client(st.secrets['CHART_DATA_API'])
    st.session_state.charts = st.session_state.gradio_client.predict(api_name="/get_all_charts")

from google import genai
from google.genai import types
client = genai.Client(api_key=st.secrets['GEMINI_API_KEY'])
thinking_model = 'gemini-2.5-flash' # nonthinking model is stupid af
price = {
    'gemini-2.5-flash': {'input': 0.3, 'output': 2.5, 'thinking': 0},
    'gemini-2.5-pro-preview-06-05': {'input': 1.25, 'output': 10, 'thinking': 0},
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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# Chat input
user_prompt = st.text_input("你想怎麼研究財經M平方的數據及圖表？", placeholder="例如：2025台灣人口結構，以圓餅圖呈現")
submit_button = st.button("✨生成圖表", type="primary")

if user_prompt and submit_button:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Reset token tracking for new request
    st.session_state.current_request_tokens = {'prompt_tokens': 0, 'candidates_tokens': 0, 'cached_content_tokens': 0, 'thoughts_tokens': 0, 'tool_use_prompt_tokens': 0, 'total_tokens': 0}

    # Loading UI
    with st.spinner("🔍 正在搜尋相關圖表..."):
        system_prompt = 'Find the most relevant chart id for the user query. Output the id.\n\n' + st.session_state.charts
        model = thinking_model
        response_type = 'application/json'
        response_schema = types.Schema(type = genai.types.Type.STRING)
        tools = None
        response = generate_content(model, user_prompt, system_prompt, response_type, response_schema, tools)
        chart_id = response.parsed
        print(chart_id)
        
        # Extract tokens from first API call
        tokens = extract_tokens(response.usage_metadata)
        for key in tokens:
            st.session_state.current_request_tokens[key] += tokens[key]

    with st.spinner("📊 正在載入圖表配置..."):
        chart_info_output, sample_series_output, series_api_output= st.session_state.gradio_client.predict(
                chart_id=chart_id,
                api_name="/get_one_chart"
        )
        chart_info = eval(chart_info_output)
        sample_series = eval(sample_series_output)
        retrieval = {
            "chart_info": chart_info,
            "sample_series": sample_series,
            "series_api": series_api_output,
            "MM Chart reference": f"[{chart_info['name_tc']}](https://www.macromicro.me/charts/{chart_id}/{chart_info['slug']})"
        }
        print(retrieval["MM Chart reference"])
        import json
        retrieval = json.dumps(retrieval, ensure_ascii=False)

    with st.spinner("🎨 正在生成圖表程式碼..."):
        system_prompt = 'Retrieval data is as below. Customized by user input, generate Highchart HTML/JS/CSS source code which calls the series API to get the complete series replacing sample series and has a button link to MM Chart reference. Write the code in multilines without code comments. Output only the Highchart HTML/JS/CSS source code.\n\n' + retrieval
        model = thinking_model
        response_type = 'application/json'
        response_schema = types.Schema(type = genai.types.Type.STRING)
        tools = None
        response = generate_content(model, user_prompt, system_prompt, response_type, response_schema, tools)
        highchart_code = response.parsed
        print(highchart_code)
        
        # Extract tokens from second API call
        tokens = extract_tokens(response.usage_metadata)
        for key in tokens:
            st.session_state.current_request_tokens[key] += tokens[key]

    st.success("✅ 圖表生成完成！（若有 🐛 就再試一次吧）")
    
    components.html(highchart_code, height=700)

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
