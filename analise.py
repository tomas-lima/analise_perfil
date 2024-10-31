import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import re

def extract_path_data(svg_content):
    # Check if the content is in table format
    if '<table' in svg_content:
        # Extract data from table
        table_regex = r'<tr><td>(.*?)</td>(.*?)</tr>'
        matches = re.findall(table_regex, svg_content, re.DOTALL)
        
        series_data = {}
        for series_name, values in matches:
            values = re.findall(r'<td>(\d+)</td>', values)
            series_data[series_name] = list(map(int, values))
        
        return series_data
    else:
        # Existing SVG path extraction logic
        path_regex = r'd="([^"]*)"'
        paths = re.findall(path_regex, svg_content)
        data_paths = [p for p in paths if len(p.split()) > 10]
        
        def parse_path(path):
            coords = path.split('L')
            coords[0] = coords[0].replace('M ', '')
            points = []
            for coord in coords:
                x, y = map(float, coord.strip().split(','))
                points.append((x, y))
            return points
        
        return [parse_path(path) for path in data_paths]

def extract_dates(svg_content):
    # Check if the content is in table format
    if '<table' in svg_content:
        # Extract timestamps from table headers
        header_regex = r'<th>(\d+)</th>'
        timestamps = re.findall(header_regex, svg_content)
        dates = [datetime.fromtimestamp(int(ts)) for ts in timestamps]
        
        return list(enumerate(dates))
    else:
        # Existing SVG date extraction logic
        date_regex = r'<text[^>]*>([A-Za-z]+ \d+)</text>'
        dates = re.findall(date_regex, svg_content)
        
        x_coord_regex = r'translate\(([^,]+),'
        x_coords = re.findall(x_coord_regex, svg_content)
        x_coords = [float(x) for x in x_coords if float(x) < 1000]
        
        date_pairs = []
        for x, date_str in zip(x_coords, dates):
            try:
                date = pd.to_datetime(f"{date_str} 2024")
                date_pairs.append((x, date))
            except:
                continue
        
        return sorted(date_pairs, key=lambda x: x[0])

def interpolate_date(x, date_points):
    """Interpolate date based on x-coordinate using actual dates from the chart"""
    x_values = np.array([x for x, _ in date_points])
    dates = [d for _, d in date_points]
    
    # Find the two closest x-coordinates
    idx = np.searchsorted(x_values, x)
    if idx == 0:
        return dates[0]
    elif idx == len(x_values):
        return dates[-1]
    
    # Linear interpolation between dates
    x1, x2 = x_values[idx-1], x_values[idx]
    d1, d2 = dates[idx-1], dates[idx]
    
    # Calculate fraction between points
    frac = (x - x1) / (x2 - x1)
    
    # Interpolate timestamp
    td = d2 - d1
    return d1 + pd.Timedelta(seconds=td.total_seconds() * frac)

def process_svg_data(svg_content):
    paths = extract_path_data(svg_content)
    date_points = extract_dates(svg_content)
    
    # Process each series
    series_data = []
    # Map extracted series names to expected names
    series_name_map = {
        'primary': 'Total Reach',
        'organic': 'Organic Reach',
        'paid': 'Ads Reach'
    }
    
    if isinstance(paths, dict):
        # If paths is a dictionary, it means we are dealing with table data
        for series_name, values in paths.items():
            mapped_name = series_name_map.get(series_name, series_name)
            for i, value in enumerate(values):
                date = date_points[i][1]
                series_data.append({
                    'date': date,
                    'value': value,
                    'series': mapped_name
                })
    else:
        # Existing SVG path processing logic
        series_names = ['Total Reach', 'Organic Reach', 'Ads Reach']
        for i, path in enumerate(paths):
            points = []
            for x, y in path:
                value = 6000000 * (1 - (y - 16) / 216.183807)
                value = max(0, value)
                date = interpolate_date(x, date_points)
                points.append({
                    'date': date,
                    'value': value,
                    'series': series_names[i]
                })
            series_data.extend(points)
    
    df = pd.DataFrame(series_data)
    df = df.sort_values(['date', 'series'])
    df_pivot = pd.pivot_table(
        df,
        index='date',
        columns='series',
        values='value',
        aggfunc='first'
    )
    
    return df_pivot.round(-2)

# Set page config
st.set_page_config(page_title="Daily Social Media Reach Analytics", layout="wide")

# Title and description
st.title("📊 Daily Social Media Reach Analytics")

st.markdown("""
This app analyzes daily reach data from social media charts. To use:
1. Copy your SVG code from the chart
2. Paste it in the text area below
3. Click 'Analyze Data' to see the results
""")

# Create a text area for SVG input
svg_input = st.text_area("Paste your SVG code here:", height=200)

# Add a button to process the SVG
if st.button("🔍 Analyze Data") and svg_input:
    try:
        with st.spinner('Processing data...'):
            df = process_svg_data(svg_input)
            
            # Create the visualization
            st.subheader("📈 Daily Reach")
            
            fig = go.Figure()
            
            # Add traces with custom colors
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['Total Reach'],
                name='Total Reach',
                line=dict(color='rgb(0, 70, 100)', width=2),
                hovertemplate='%{x|%b %d, %Y}<br>%{y:,.0f}'
            ))
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['Organic Reach'],
                name='Organic Reach',
                line=dict(color='rgb(30, 144, 255)', width=2),
                hovertemplate='%{x|%b %d, %Y}<br>%{y:,.0f}'
            ))
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['Ads Reach'],
                name='Ads Reach',
                line=dict(color='rgb(173, 216, 230)', width=2),
                hovertemplate='%{x|%b %d, %Y}<br>%{y:,.0f}'
            ))
            
            fig.update_layout(
                height=500,
                xaxis_title="Date",
                yaxis_title="Daily Reach",
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                yaxis=dict(
                    tickformat=',d',
                    rangemode='tozero'
                ),
                margin=dict(l=0, r=0, t=20, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create columns for key metrics
            col1, col2, col3 = st.columns(3)
            
            # Calculate peak and average metrics
            with col1:
                st.metric(
                    "Peak Total Reach", 
                    f"{df['Total Reach'].max():,.0f}",
                    f"Avg: {df['Total Reach'].mean():,.0f}"
                )
            
            with col2:
                st.metric(
                    "Peak Organic Reach",
                    f"{df['Organic Reach'].max():,.0f}",
                    f"Avg: {df['Organic Reach'].mean():,.0f}"
                )
            
            with col3:
                st.metric(
                    "Peak Ads Reach",
                    f"{df['Ads Reach'].max():,.0f}",
                    f"Avg: {df['Ads Reach'].mean():,.0f}"
                )
            
            # Add date selector for detailed view
            st.subheader("📋 Daily Data Explorer")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", df.index.min())
            with col2:
                end_date = st.date_input("End Date", df.index.max())
            
            # Filter data based on selected dates
            mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            filtered_df = df.loc[mask]
            
            # Show the filtered data table
            st.dataframe(
                filtered_df.style.format("{:,.0f}")
                          .background_gradient(cmap='Blues', subset=['Total Reach'])
            )
            
            # Add download button for CSV
            csv = df.to_csv()
            st.download_button(
                label="💾 Download Complete Daily Data as CSV",
                data=csv,
                file_name="daily_reach_data.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"⚠️ Error processing data: {str(e)}")
        st.error("Please check if the SVG code is valid and contains the expected data format.")

# Add footer with instructions
st.markdown("---")
st.markdown("""
### 📝 Notes:
- Shows daily reach metrics for Total, Organic, and Ads reach
- Date range is based on actual chart labels
- Use the date selector to explore specific time periods
- Download option provides the complete daily dataset
""")

'''
cd "C:\Users\tomas\OneDrive\Área de Trabalho\analise_perfil" && python -m venv venv && .\venv\Scripts\activate && streamlit run analise.py
'''

