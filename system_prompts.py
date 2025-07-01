csv_only = """
# You are tasked with coding a Python module that plots a Plotly chart. Your module should:
- Read and parse uploaded_series.csv with the following logic:
```
import pandas as pd
import os
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    try:
        dfs = []
        if os.path.exists('uploaded_series.csv'):
            df = pd.read_csv('uploaded_series.csv')
            df.index = pd.to_datetime(df.iloc[:,0])
            for column in df.columns[1:]:
                dfs.append(df[column])
        else:
            st.warning("No uploaded CSV file found.")
            return

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        trace_count = 0
        all_dates = pd.to_datetime([])
        for i, series in enumerate(dfs):
            fig.add_trace(
                go.Scatter(x=series.index, y=series, name=series.name),
                secondary_y=trace_count % 2 != 0
            )
            trace_count += 1
            all_dates = all_dates.union(series.index)

        fig.update_layout(
            title_text='Custom Chart from Uploaded Data',
            template='plotly_dark',
            legend_title_text='Series',
            yaxis=dict(autorange=True),
            yaxis2=dict(autorange=True)
        )
        fig.update_xaxes(title_text='Date', range=[all_dates.min(), all_dates.max()])
        fig.update_yaxes(title_text="Primary Y-Axis", secondary_y=False)
        fig.update_yaxes(title_text="Secondary Y-Axis", secondary_y=True)

        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred: {e}")

```
- Handle different data ranges by using secondary y-axis when ranges differ significantly.
- Provide series color selection options.
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
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    try:
        dfs = []
        r = requests.get(series_api)
        r.raise_for_status()  # Raise an exception for bad status codes
        fetched_series = r.json()
        for series, name in zip(fetched_series, series_names):
            df = pd.DataFrame(series)
            df.index = pd.to_datetime(df[0])
            dfs.append(pd.DataFrame({name: df[1]}))

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        trace_count = 0
        all_dates = pd.to_datetime([])
        for i, df_series in enumerate(dfs):
            for col in df_series.columns:
                fig.add_trace(
                    go.Scatter(x=df_series.index, y=df_series[col], name=col),
                    secondary_y=trace_count % 2 != 0
                )
                trace_count += 1
                all_dates = all_dates.union(df_series.index)

        fig.update_layout(
            title_text='MacroMicro Data Chart',
            template='plotly_dark',
            legend_title_text='Series',
            yaxis=dict(autorange=True),
            yaxis2=dict(autorange=True)
        )
        fig.update_xaxes(title_text='Date', range=[all_dates.min(), all_dates.max()])
        fig.update_yaxes(title_text="Primary Y-Axis", secondary_y=False)
        fig.update_yaxes(title_text="Secondary Y-Axis", secondary_y=True)

        st.plotly_chart(fig)
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from API: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
```
- Customize the chart based on user input text prompt.
- Handle different data ranges by using secondary y-axis when ranges differ significantly.
- Provide series color selection options.
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
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    try:
        dfs = []
        # Fetch data from API
        r = requests.get(series_api)
        r.raise_for_status()
        fetched_series = r.json()
        for series, name in zip(fetched_series, series_names):
            df = pd.DataFrame(series)
            df.index = pd.to_datetime(df[0])
            dfs.append(pd.DataFrame({name: df[1]}))

        # Read data from CSV
        if os.path.exists('uploaded_series.csv'):
            df_csv = pd.read_csv('uploaded_series.csv')
            df_csv.index = pd.to_datetime(df_csv.iloc[:,0])
            for column in df_csv.columns[1:]:
                dfs.append(df_csv[column])
        else:
            st.warning("No uploaded CSV file found.")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        trace_count = 0
        all_dates = pd.to_datetime([])
        for i, series in enumerate(dfs):
            if isinstance(series, pd.DataFrame):
                 for col in series.columns:
                    fig.add_trace(
                        go.Scatter(x=series.index, y=series[col], name=col),
                        secondary_y=trace_count % 2 != 0
                    )
                    trace_count += 1
                    all_dates = all_dates.union(series.index)
            else: # it's a series
                fig.add_trace(
                    go.Scatter(x=series.index, y=series, name=series.name),
                    secondary_y=trace_count % 2 != 0
                )
                trace_count += 1
                all_dates = all_dates.union(series.index)


        fig.update_layout(
            title_text='MacroMicro and Uploaded Data Chart',
            template='plotly_dark',
            legend_title_text='Series',
            yaxis=dict(autorange=True),
            yaxis2=dict(autorange=True)
        )
        fig.update_xaxes(title_text='Date', range=[all_dates.min(), all_dates.max()])
        fig.update_yaxes(title_text="Primary Y-Axis", secondary_y=False)
        fig.update_yaxes(title_text="Secondary Y-Axis", secondary_y=True)

        st.plotly_chart(fig)
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from API: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

```
- Customize the chart based on user input text prompt.
- Handle different data ranges by using secondary y-axis when ranges differ significantly.
- Provide series color selection options.
- Use streamlit.plotly_chart(fig) to display the chart in main() function.
- Ensure plotly_module can be imported and plotly_module.main() can be called to run the chart.
- Do not write code comments.
"""