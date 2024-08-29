import dash
from dash import dcc, html, Input, Output, State, callback_context
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import argparse
import os

# Parse the arguments
parser = argparse.ArgumentParser(description='Visualize the PathBench results')
parser.add_argument('--results', '-r', type=str, required=True, help='Path to the results in the experiment directory')
args = parser.parse_args()

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load all aggregated CSV files in the directory
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

# Global variable to track dynamically created columns
created_columns = []

def group_by_x_axis(df, x_axis, y_axis):
    global created_columns
    
    if x_axis and y_axis:
        # Create a new column for the mean of the y_axis values grouped by x_axis
        new_column_name = f'{y_axis}_mean'
        df[new_column_name] = df.groupby(x_axis)[y_axis].transform('mean')
        
        # Sort the DataFrame by the new mean column in descending order and the x_axis in ascending order
        sorted_df = df.sort_values(by=[new_column_name, x_axis], ascending=[False, True]).reset_index(drop=True)
        
        # Keep track of the created column
        created_columns.append(new_column_name)
        
        return sorted_df, new_column_name
    return df, y_axis

# Define layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Img(src='../../PathBench-logo-gecentreerd.png', height="100px"), width="auto", style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
        dbc.Col(html.H1("PathBench Visualization Dashboard"), className="text-center mt-4 mb-4", width="auto")
    ], justify="center"),
    
    # Add a Row with some space before the options menu
    dbc.Row([
        dbc.Col(width=12, style={"margin-bottom": "20px"})  # Adding space
    ]),

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
    
    # Add a Row with some space before the plotly plot
    dbc.Row([
        dbc.Col(width=12, style={"margin-bottom": "10px"})  # Adding space
    ]),
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
    'backgroundColor': '#f8f9fa',  # Light grey background color
    'minHeight': '100vh',  # Ensure the background covers the full height
})

# Combined callback for updating options, values, and the plot
@app.callback(
    [Output('x-axis', 'options'),
     Output('y-axis', 'options'),
     Output('heatmap-value', 'options'),
     Output('heatmap-column', 'style'),
     Output('plot', 'figure'),
     Output('plot', 'style'),
     Output('x-axis', 'value'),
     Output('y-axis', 'value'),
     Output('heatmap-value', 'value')],
    [Input('dataset-selection', 'value'),
     Input('plot-type', 'value'),
     Input('x-axis', 'value'),
     Input('y-axis', 'value'),
     Input('heatmap-value', 'value'),
     Input('group-button', 'n_clicks'),
     Input('reset-button', 'n_clicks')],
    [State('x-axis', 'value'),
     State('y-axis', 'value')]
)
def update_layout_and_plot(dataset_selection, plot_type, x_axis, y_axis, heatmap_value, n_clicks_group, n_clicks_reset, x_axis_state, y_axis_state):
    global created_columns
    
    df = df_dict[dataset_selection]

    # Filter out created columns from the y-axis options
    y_axis_options = [{'label': col, 'value': col} for col in df.columns if col not in created_columns]
    x_axis_options = [{'label': col, 'value': col} for col in df.columns if col not in created_columns]
    heatmap_value_options = [{'label': col, 'value': col} for col in df.columns if col not in created_columns]

    # Determine the trigger of the callback
    triggered = callback_context.triggered[0]['prop_id'].split('.')[0]

    # Handle reset button
    if triggered == 'reset-button':
        return (x_axis_options, y_axis_options, heatmap_value_options, {'display': 'none'}, {}, {'height': '600px'},
                None, None, None)

    # Show heatmap column selector only when heatmap is selected
    heatmap_style = {'display': 'block'} if plot_type == 'heatmap' else {'display': 'none'}

    # Group by x-axis and calculate mean of y-axis if group button is clicked
    if triggered == 'group-button' and x_axis_state and y_axis_state:
        df, y_axis = group_by_x_axis(df, x_axis_state, y_axis_state)

    # Specify columns to include in hover data
    hover_columns = ['tile_px', 'tile_um', 'normalization', 'feature_extraction', 'mil', 'loss', 'activation_function']
    hover_data = {col: True for col in hover_columns if col in df.columns}
    
    # Determine if legend_text can be used for color
    color_column = 'legend_text' if 'legend_text' in df.columns else None
    
    # Generate the plot based on the plot type
    if plot_type == 'scatter' and x_axis and y_axis:
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_column,
                         title=f'Scatter Plot of {y_axis} vs {x_axis}', 
                         template='plotly_white', hover_data=hover_data)
        
    elif plot_type == 'histogram' and x_axis and y_axis:
        # Using Histogram to plot the sorted data
        fig = px.histogram(df, x=x_axis, y=y_axis, color=x_axis, text_auto=True,
                        title=f'Histogram of {y_axis} vs {x_axis}', 
                        template='plotly_white', hover_data=hover_data,
                        histfunc='avg')  # Set histfunc to sum to use the sorted means directly
        
        # Format the text to show only three decimal points
        fig.update_traces(texttemplate='%{y:.3f}')
        
        # Set barmode to overlay or group
        fig.update_layout(barmode='overlay')

        # Adjust the x-axis ordering to reflect the sorted data
        fig.update_xaxes(categoryorder='total descending')

    elif plot_type == 'bar' and x_axis and y_axis:
        fig = px.bar(df, x=x_axis, y=y_axis, color=color_column, 
                     title=f'Bar Plot of {y_axis} vs {x_axis}', 
                     template='plotly_white', hover_data=hover_data,
                     barmode='group')
        fig.update_layout(xaxis=dict(categoryorder='category ascending'))
    elif plot_type == 'heatmap' and x_axis and y_axis and heatmap_value:
        agg_df = df.groupby([x_axis, y_axis]).agg({heatmap_value: 'mean'}).reset_index()
        fig = px.density_heatmap(agg_df, x=x_axis, y=y_axis, z=heatmap_value, 
                                 color_continuous_scale='Viridis', 
                                 title=f'Heatmap of Mean {heatmap_value} with {y_axis} vs {x_axis}', 
                                 template='plotly_white')

        # Add text annotations with three decimal places
        fig.update_traces(texttemplate='%{z:.3f}')
        fig.update_layout(annotations=[
            dict(
                x=row[x_axis],
                y=row[y_axis],
                text=f'{row[heatmap_value]:.3f}',
                showarrow=False,
                font=dict(color="white" if row[heatmap_value] < agg_df[heatmap_value].mean() else "black")
            ) for _, row in agg_df.iterrows()
        ])
        
    elif plot_type == 'box' and x_axis and y_axis:
        # Ensure a single boxplot per x-axis group
        fig = px.box(df, x=x_axis, y=y_axis, color=x_axis, 
                     title=f'Box Plot of {y_axis} grouped by {x_axis}', 
                     template='plotly_white', hover_data=hover_data)
    elif plot_type == 'strip' and x_axis and y_axis:
        fig = px.strip(df, x=x_axis, y=y_axis, color=color_column, 
                       title=f'Strip Plot of {y_axis} vs {x_axis}', 
                       template='plotly_white', hover_data=hover_data)
    else:
        fig = px.scatter(title="Please select X-axis and Y-axis values")

    # Dynamically adjust the legend text size
    fig.update_layout(
        legend_title_text='Legend',
        legend=dict(font=dict(size=10), itemwidth=30),  # Adjust legend text size here
        height=600,  # Fix the plot height
        showlegend=True,  # Ensure legend is shown
        plot_bgcolor='#f8f9fa',  # Light grey background color
        paper_bgcolor='#f8f9fa',  # Light grey background color
    )
    
    return (x_axis_options, y_axis_options, heatmap_value_options, heatmap_style, fig, {'height': '600px'},
            x_axis, y_axis, heatmap_value)

if __name__ == '__main__':
    app.run_server(debug=True)