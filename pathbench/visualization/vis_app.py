import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import os
import argparse

# Parse the arguments
parser = argparse.ArgumentParser(description='Visualize the PathBench results')
parser.add_argument('--results', '-r', type=str, required=True, help='Path to the results in the experiment directory')
args = parser.parse_args()

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load your data (example data loaded here, replace with your data)
df_dict = {}
for file in os.listdir(args.results):
    if file.endswith(".csv") and 'agg' in file:
        df = pd.read_csv(os.path.join(args.results, file))
        filename = file.split('.')[0]
        df_dict[filename] = df

# Function to concatenate specified columns for legend text
def create_legend_text(df):
    columns = ['tile_px', 'tile_um', 'normalization', 'feature_extraction', 'mil', 'loss', 'activation_function']
    existing_columns = [col for col in columns if col in df.columns]
    if existing_columns:
        df['legend_text'] = df.apply(lambda row: ", ".join([str(row[col]) for col in existing_columns]), axis=1)
    else:
        df['legend_text'] = "N/A"
    return df

# Apply the function to all DataFrames in df_dict
for key, df in df_dict.items():
    df_dict[key] = create_legend_text(df)

def generate_filter_dropdowns(df):
    filter_dropdowns = []
    for column in df.columns:
        # Check if the column is either categorical or specifically 'tile_px'
        if (column == 'tile_px' or df[column].dtype == 'object' or pd.api.types.is_categorical_dtype(df[column])) and column != 'legend_text':
            unique_values = df[column].dropna().unique()  # Ensure to remove NaN values
            if len(unique_values) > 0:
                filter_dropdowns.append(
                    dbc.Col([
                        html.Label(column, className='font-weight-bold', style={'font-size': '12px'}),
                        dcc.Dropdown(
                            id={'type': 'filter-dropdown', 'index': column},
                            options=[{'label': str(val), 'value': val} for val in unique_values],
                            multi=True,
                            clearable=True,
                            placeholder=f'Select {column} values',
                            className='mb-3',
                            style={'font-size': '12px'}
                        )
                    ], width=3)  # Adjust the width here to make the columns smaller
                )
    return filter_dropdowns

# Global variable to track dynamically created columns
created_columns = []

def group_by_x_axis(df, x_axis, y_axis):
    global created_columns
    
    if x_axis and y_axis:
        # Create a new column for the mean of the y_axis values grouped by the x_axis with respect to the filtered data
        new_column_name = f'{y_axis}_mean_grouped_by_{x_axis}'
        df[new_column_name] = df.groupby(x_axis)[y_axis].transform('mean')
        
        # Sort the DataFrame by the new mean column in descending order and the x_axis in ascending order
        sorted_df = df.sort_values(by=[new_column_name, x_axis], ascending=[False, True]).reset_index(drop=True)
        
        # Keep track of the created column to avoid reusing it in y-axis options
        if new_column_name not in created_columns:
            created_columns.append(new_column_name)
        
        return sorted_df, new_column_name
    
    return df, y_axis

# Define layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("PathBench Visualization Dashboard"), className="text-center mt-4 mb-4")
    ], justify="center"),
    
    dbc.Row([
        dbc.Col([
            html.Label('Select Dataset', className='font-weight-bold'),
            dcc.Dropdown(
                id='dataset-selection',
                options=[{'label': key, 'value': key} for key in df_dict.keys()],
                value=list(df_dict.keys())[0],  # Set default value to the first dataset
                clearable=False,
                className='mb-3'
            )
        ], width=3, className='mb-4'),

        dbc.Col([
            html.Label('Plot Type', className='font-weight-bold'),
            dcc.Dropdown(
                id='plot-type',
                options=[
                    {'label': 'Scatter Plot', 'value': 'scatter'},
                    {'label': 'Histogram', 'value': 'histogram'},
                    {'label': 'Heatmap', 'value': 'heatmap'},
                    {'label': 'Box Plot', 'value': 'box'},
                    {'label': 'Strip Plot', 'value': 'strip'}
                ],
                value='scatter',
                clearable=False,
                className='mb-3'
            )
        ], width=3, className='mb-4')
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label('X-axis', className='font-weight-bold'),
            dcc.Dropdown(
                id='x-axis',
                clearable=False,
                className='mb-3'
            )
        ], width=6),
        dbc.Col([
            html.Label('Y-axis', className='font-weight-bold'),
            dcc.Dropdown(
                id='y-axis',
                clearable=False,
                className='mb-3'
            )
        ], width=6)
    ]),

    dbc.Row([
        dbc.Col([
            html.Label('Filters', className='font-weight-bold'),
        ], width=12)
    ]),
    
    dbc.Row(id='filter-dropdowns'),

    dbc.Row([
        dbc.Col([
            html.Label('Heatmap Value', className='font-weight-bold'),
            dcc.Dropdown(
                id='heatmap-value',
                clearable=True,
                className='mb-3'
            )
        ], width=6, id='heatmap-column', style={'display': 'none'})
    ]),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='plot', style={'height': '600px',
                                            'background-color': '#f8f9fa'}), width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Button("Group by X-axis", id="group-button", color="primary", className="mr-2", n_clicks=0),
            dbc.Button("Reset", id="reset-button", color="secondary", className="mr-2", n_clicks=0)
        ], width=12, className="text-center mt-4")
    ])
], fluid=True, style={
    'padding': '40px',
    'backgroundColor': '#f8f9fa',
    'minHeight': '100vh',
})

# Callback to update filter dropdowns dynamically
@app.callback(
    Output('filter-dropdowns', 'children'),
    Input('dataset-selection', 'value')
)
def update_filter_dropdowns(dataset_selection):
    df = df_dict[dataset_selection]
    return generate_filter_dropdowns(df)

# Callback to update x-axis and y-axis dropdown options based on selected dataset
@app.callback(
    [Output('x-axis', 'options'),
     Output('y-axis', 'options'),
     Output('x-axis', 'value'),
     Output('y-axis', 'value')],
    Input('dataset-selection', 'value')
)
def update_axis_options(dataset_selection):
    df = df_dict[dataset_selection]
    options = [{'label': col, 'value': col} for col in df.columns if col not in ['legend_text']]
    
    return options, options, None, None

# Combined callback for updating the plot
@app.callback(
    [Output('heatmap-value', 'options'),
     Output('heatmap-column', 'style'),
     Output('plot', 'figure')],
    [Input('dataset-selection', 'value'),
     Input('plot-type', 'value'),
     Input('x-axis', 'value'),
     Input('y-axis', 'value'),
     Input('heatmap-value', 'value'),
     Input('group-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input({'type': 'filter-dropdown', 'index': ALL}, 'value')],
    [State('x-axis', 'value'),
     State('y-axis', 'value')]
)
def update_layout_and_plot(dataset_selection, plot_type, x_axis, y_axis, heatmap_value, n_clicks_group, n_clicks_reset,
                           filter_values, x_axis_state, y_axis_state):
    global created_columns
    
    df = df_dict[dataset_selection]

    # Construct filter dictionary from the inputs, safely getting the 'index' key
    filter_values_dict = {}
    for filter_id, val in zip(callback_context.inputs_list[7], filter_values):
        filter_index = filter_id.get('id', {}).get('index')  # Safely retrieve the 'index'
        if filter_index:
            filter_values_dict[filter_index] = val

    # Apply filters to exclude selected values
    for col, filter_val in filter_values_dict.items():
        if filter_val:
            df = df[~df[col].isin(filter_val)]

    if df.empty:
        return ([], {'display': 'none'}, px.scatter(title="No Data Available"))

    # Determine the trigger of the callback
    triggered = callback_context.triggered[0]['prop_id'].split('.')[0]

    # Grouping logic
    if triggered == 'group-button' and x_axis_state and y_axis_state:
        df, y_axis = group_by_x_axis(df, x_axis_state, y_axis_state)

    # Check if y-axis is numeric:
    if y_axis and df[y_axis].dtype == 'object':
        df = df.sort_values(by=y_axis, ascending=False)

    # Filter out created columns from the heatmap options
    heatmap_value_options = [{'label': col, 'value': col} for col in df.columns if col not in created_columns]

    # Show heatmap column selector only when heatmap is selected
    heatmap_style = {'display': 'block'} if plot_type == 'heatmap' else {'display': 'none'}

    # Generate the plot based on the plot type
    if not x_axis or not y_axis:
        fig = px.scatter(title="Please select X-axis and Y-axis values")
    elif plot_type == 'scatter':
        fig = px.scatter(df, x=x_axis, y=y_axis, color='legend_text' if 'legend_text' in df.columns else None,
                         title=f'Scatter Plot of {y_axis} vs {x_axis}', template='plotly_white')
        
    elif plot_type == 'histogram':
        fig = px.histogram(df, x=x_axis, y=y_axis, color=x_axis, text_auto=True,
                           title=f'Histogram of {y_axis} vs {x_axis}', template='plotly_white',
                           histfunc='avg' if y_axis else 'count')
        
    elif plot_type == 'box':
        fig = px.box(df, x=x_axis, y=y_axis, color=x_axis, title=f'Box Plot of {y_axis} grouped by {x_axis}', 
                     template='plotly_white')
        
    elif plot_type == 'strip':
        fig = px.strip(df, x=x_axis, y=y_axis, color='legend_text' if 'legend_text' in df.columns else None,
                       title=f'Strip Plot of {y_axis} vs {x_axis}', template='plotly_white')
        
    elif plot_type == 'heatmap' and heatmap_value:
        agg_df = df.groupby([x_axis, y_axis]).agg({heatmap_value: 'mean'}).reset_index()
        fig = px.density_heatmap(agg_df, x=x_axis, y=y_axis, z=heatmap_value, 
                                 color_continuous_scale='Viridis', title=f'Heatmap of Mean {heatmap_value}', 
                                 template='plotly_white')

    fig.update_layout(
        height=600,
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa',
    )
    
    return heatmap_value_options, heatmap_style, fig
if __name__ == '__main__':
    app.run_server(debug=True)
