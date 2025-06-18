import streamlit as st
import streamlit.components.v1 as components

from gradio_client import Client
gradio_client = Client(st.secrets['CHART_DATA_API'])
if "charts" not in st.session_state:
    st.session_state.charts = gradio_client.predict(api_name="/get_all_charts")

price = {
    'gemini-2.0-flash': {'input': 0.1, 'output': 0.4, 'thinking': 0},
    'gemini-2.5-flash-preview-05-20': {'input': 0.15, 'output': 0.6, 'thinking': 3.5},
    'gemini-2.5-pro-preview-06-05': {'input': 1.25, 'output': 10, 'thinking': 0},
}

def extract_tokens(usage_metadata):
    try:
        print(f"DEBUG: usage_metadata = {usage_metadata}")
        print(f"DEBUG: usage_metadata type = {type(usage_metadata)}")
        
        prompt_tokens = 0
        candidates_tokens = 0
        cached_tokens = 0
        thoughts_tokens = 0
        total_tokens = 0
        
        # Check all attributes and print for debugging
        for attr in dir(usage_metadata):
            if not attr.startswith('_'):
                value = getattr(usage_metadata, attr, 0)
                print(f"DEBUG: {attr} = {value}")
                
                if 'prompt' in attr.lower() and 'token' in attr.lower():
                    prompt_tokens = value or 0
                elif 'candidate' in attr.lower() and 'token' in attr.lower():
                    candidates_tokens = value or 0
                elif 'cached' in attr.lower() and 'token' in attr.lower():
                    cached_tokens = value or 0
                elif 'thought' in attr.lower() and 'token' in attr.lower():
                    thoughts_tokens = value or 0
                elif 'total' in attr.lower() and 'token' in attr.lower():
                    total_tokens = value or 0
        
        # Fallback: try common attribute names directly
        if prompt_tokens == 0:
            prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0) or getattr(usage_metadata, 'input_tokens', 0)
        if candidates_tokens == 0:
            candidates_tokens = getattr(usage_metadata, 'candidates_token_count', 0) or getattr(usage_metadata, 'output_tokens', 0)
        if total_tokens == 0:
            total_tokens = getattr(usage_metadata, 'total_token_count', 0) or (prompt_tokens + candidates_tokens + thoughts_tokens)
        
        result = {
            'prompt_tokens': prompt_tokens,
            'candidates_tokens': candidates_tokens,
            'cached_tokens': cached_tokens,
            'thoughts_tokens': thoughts_tokens,
            'total_tokens': total_tokens
        }
        
        print(f"DEBUG: Extracted tokens = {result}")
        return result
        
    except Exception as e:
        print(f"ERROR extracting tokens: {e}")
        return {'prompt_tokens': 0, 'candidates_tokens': 0, 'cached_tokens': 0, 'thoughts_tokens': 0, 'total_tokens': 0}
def calculate_cost(tokens, model_name):
    return round((tokens['prompt_tokens'] * price[model_name]['input'] + tokens['candidates_tokens'] * price[model_name]['output'] + tokens['thoughts_tokens'] * price[model_name]['thinking'])/1e6, 3)

from google import genai
from google.genai import types
client = genai.Client(api_key=st.secrets['GEMINI_API_KEY'])
non_thinking_model = 'gemini-2.0-flash'
thinking_model = 'gemini-2.5-flash-preview-05-20'
response_type = 'application/json'
response_schema = str
def generate_content(model, user_prompt, system_prompt, response_type, response_schema):
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type=response_type,
            response_schema=response_schema,
            # tools=tools,
        )
    )
    return response

st.title("MM AI Charting ![](https://cdn.macromicro.me/assets/img/favicons/favicon-32.png)✨📈")

# Initialize session state for current request tokens
if "current_request_tokens" not in st.session_state:
    st.session_state.current_request_tokens = {'prompt_tokens': 0, 'candidates_tokens': 0, 'cached_tokens': 0, 'thoughts_tokens': 0, 'total_tokens': 0}

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# Chat input
user_prompt = st.text_input("輸入你想研究的財經M平方圖表名稱或關鍵字", placeholder="例如：小台散戶、柯博文指數、美債殖利率vs基準利率")
submit_button = st.button("✨生成圖表", type="primary", disabled=not user_prompt.strip())

if user_prompt and submit_button:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Reset token tracking for new request
    st.session_state.current_request_tokens = {'prompt_tokens': 0, 'candidates_tokens': 0, 'cached_tokens': 0, 'thoughts_tokens': 0, 'total_tokens': 0}

    # Loading UI
    with st.spinner("🔍 正在搜尋相關圖表..."):
        system_prompt = 'Find the most relevant chart id for the user query. Output the id.\n\n' + st.session_state.charts
        model = thinking_model
        response = generate_content(model, user_prompt, system_prompt, response_type, response_schema)
        response_parsed = response.parsed
        print(response_parsed)
        
        # Extract tokens from first API call
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens = extract_tokens(response.usage_metadata)
            for key in st.session_state.current_request_tokens:
                st.session_state.current_request_tokens[key] += tokens[key]

    with st.spinner("📊 正在載入圖表配置..."):
        chart_info_output, sample_series_output, series_api_output= gradio_client.predict(
                chart_id=response_parsed,
                api_name="/get_one_chart"
        )
        import json
        user_prompt = {
            "chart_info": eval(chart_info_output),
            "sample_series": eval(sample_series_output),
            "series_api": series_api_output
        }
        user_prompt = json.dumps(user_prompt, ensure_ascii=False)
        print(user_prompt)

    with st.spinner("🎨 正在生成圖表程式碼..."):
        system_prompt = 'Given chart_info and sample_series, generate Highchart HTML/JS/CSS source code which calls the series API to get the complete series data from the frontend. Output only the Highchart HTML/JS/CSS source code.'
        model = thinking_model
        response = generate_content(model, user_prompt, system_prompt, response_type, response_schema)
        response_parsed = response.parsed
        print(response_parsed)
        
        # Extract tokens from second API call
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens = extract_tokens(response.usage_metadata)
            for key in st.session_state.current_request_tokens:
                st.session_state.current_request_tokens[key] += tokens[key]

    st.success("✅ 圖表生成完成！（若有 🐛 就再試一次吧）")
    
    st.components.v1.html(response_parsed, height=800)

# Display token count and cost in sidebar
with st.sidebar:
    st.header("✨")
    
    # Display token counts
    st.metric("Total Tokens", st.session_state.current_request_tokens['total_tokens'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Input Tokens", st.session_state.current_request_tokens['prompt_tokens'])
        st.metric("Cached Tokens", st.session_state.current_request_tokens['cached_tokens'])
    with col2:
        st.metric("Output Tokens", st.session_state.current_request_tokens['candidates_tokens'])
        st.metric("Thinking Tokens", st.session_state.current_request_tokens['thoughts_tokens'])
    
    # Display cost
    if st.session_state.current_request_tokens['total_tokens'] > 0:
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
