csv_only = """
# You are tasked with coding a Python module that plots a Plotly chart. Your module should:
- Read and parse uploaded_series.csv with the following logic:
```
import pandas as pd
import os
def main():
    dfs = []
    if os.path.exists('uploaded_series.csv'):
        df = pd.read_csv('uploaded_series.csv')
        df.index = pd.to_datetime(df.iloc[:,0])
        for column in df.columns[1:]:
            dfs.append(df[column])
    else:
        st.warning("No uploaded CSV file found.")
```
- Handle different data ranges by using secondary y-axis when ranges differ significantly. Use log scale for y-axis if any series has a large range.
- Use streamlit.plotly_chart(fig) to display the chart in main() function.
- Ensure plotly_module can be imported and plotly_module.main() can be called to run the chart.
- Do not write code comments.
"""

api_only = """
# You are tasked with coding a Python module that plots a Plotly chart. Your module should:
- Read and parse fetched_series from series_api with the following logic:
```
import requests
import pandas as pd
def main():
    dfs = []
    r = requests.get(series_api)
    fetched_series = r.json()
    for series, name in zip(fetched_series, series_names):
        df = pd.DataFrame(series)
        df.index = pd.to_datetime(df[0])
        dfs.append(pd.DataFrame({name: df[1]}))
```
- Customize the chart based on user input text prompt.
- Handle different data ranges by using secondary y-axis when ranges differ significantly. Use log scale for y-axis if any series has a large range.
- Use streamlit.plotly_chart(fig) to display the chart in main() function.
- Ensure plotly_module can be imported and plotly_module.main() can be called to run the chart.
- Do not write code comments.
"""

api_csv = """
# You are tasked with coding a Python module that plots a Plotly chart. Your module should:
- Read and parse fetched_series from series_api and uploaded_series.csv with the following logic:
```
import requests
import pandas as pd
import os
def main():
    dfs = []
    r = requests.get(series_api)
    fetched_series = r.json()
    for series, name in zip(fetched_series, series_names):
        df = pd.DataFrame(series)
        df.index = pd.to_datetime(df[0])
        dfs.append(pd.DataFrame({name: df[1]}))

    if os.path.exists('uploaded_series.csv'):
        df = pd.read_csv('uploaded_series.csv')
        df.index = pd.to_datetime(df.iloc[:,0])
        for column in df.columns[1:]:
            dfs.append(df[column])
    else:
        st.warning("No uploaded CSV file found.")
```
- Customize the chart based on user input text prompt.
- Handle different data ranges by using secondary y-axis when ranges differ significantly. Use log scale for y-axis if any series has a large range.
- Use streamlit.plotly_chart(fig) to display the chart in main() function.
- Ensure plotly_module can be imported and plotly_module.main() can be called to run the chart.
- Do not write code comments.
"""