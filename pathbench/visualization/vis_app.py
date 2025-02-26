import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import os
import argparse

# ---------------------------
# Parse Command Line Arguments
# ---------------------------
parser = argparse.ArgumentParser(description='Visualize the PathBench results')
parser.add_argument('--results', '-r', type=str, required=True,
                    help='Path to the results in the experiment directory')
args = parser.parse_args()

# ---------------------------
# Initialize the Dash Application
# ---------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# ---------------------------
# Load DataFrames from CSV files
# ---------------------------
df_dict = {}
for file in os.listdir(args.results):
    # Only load CSV files that have 'agg' in the filename
    if file.endswith(".csv") and 'agg' in file:
        df = pd.read_csv(os.path.join(args.results, file))
        filename = file.split('.')[0]
        df_dict[filename] = df

# ---------------------------
# Helper Function: Create Legend Text
# ---------------------------
def create_legend_text(df):
    """
    Create a new column 'legend_text' by concatenating specific columns
    to use for color grouping in plots.
    """
    columns = ['tile_px', 'tile_um', 'normalization', 'feature_extraction', 'mil', 'loss', 'activation_function']
    existing_columns = [col for col in columns if col in df.columns]
    if existing_columns:
        df['legend_text'] = df.apply(lambda row: ", ".join([str(row[col]) for col in existing_columns]), axis=1)
    else:
        df['legend_text'] = "N/A"
    return df

# Apply legend text creation to every DataFrame
for key, df in df_dict.items():
    df_dict[key] = create_legend_text(df)

# ---------------------------
# Helper Function: Generate Filter Dropdowns
# ---------------------------
def generate_filter_dropdowns(df):
    """
    Generate a list of dropdown filters for each categorical column
    (and for 'tile_px') in the DataFrame, except the legend_text column.
    """
    filter_dropdowns = []
    for column in df.columns:
        # If column is either 'tile_px' or an object/categorical type and not 'legend_text'
        if (column == 'tile_px' or df[column].dtype == 'object' or pd.api.types.is_categorical_dtype(df[column])) and column != 'legend_text':
            unique_values = df[column].dropna().unique()  # Remove NaN values
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
                    ], width=3)
                )
    return filter_dropdowns

# Global list to track dynamically created columns (used in grouping)
created_columns = []

# ---------------------------
# Helper Function: Group Data by X-axis
# ---------------------------
def group_by_x_axis(df, x_axis, y_axis):
    """
    Group the DataFrame by x_axis and compute the mean of y_axis.
    A new column is created with the name '{y_axis}_mean_grouped_by_{x_axis}'.

    Returns:
        (sorted_df, new_column_name)
    """
    global created_columns
    if x_axis and y_axis:
        new_column_name = f'{y_axis}_mean_grouped_by_{x_axis}'
        # Create the new column as the mean of y_axis for each group defined by x_axis
        df[new_column_name] = df.groupby(x_axis)[y_axis].transform('mean')

        # Sort by the new column (descending) and x_axis (ascending)
        sorted_df = df.sort_values(by=[new_column_name, x_axis], ascending=[False, True]).reset_index(drop=True)

        # Record the newly created column to exclude it from heatmap options
        if new_column_name not in created_columns:
            created_columns.append(new_column_name)

        return sorted_df, new_column_name
    return df, y_axis

# ---------------------------
# Define the App Layout
# ---------------------------
app.layout = dbc.Container([
    # Header: Logo and Title side-by-side
    dbc.Row(
        dbc.Col(
            html.Div([
                html.Img(
                    src='../../thumbnail_PathBench-logo-horizontaal.png',
                    height="100px",
                    style={'margin': '20px'}
                ),
                html.H1(
                    "Visualization Dashboard",
                    style={'marginLeft': '20px', 'margin': '20px 0', 'fontSize': '3rem', 'fontFamily': 'Arial',
                           'paddingTop' : '10px', 'color': '#ef27f2'}
                )
            ],
            style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center'
            })
        )
    ),

    # Dataset and plot type selection
    dbc.Row([
        dbc.Col([
            html.Label('Select Dataset', className='font-weight-bold'),
            dcc.Dropdown(
                id='dataset-selection',
                options=[{'label': key, 'value': key} for key in df_dict.keys()],
                value=list(df_dict.keys())[0],  # Default to the first dataset
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

    # X-axis and Y-axis selection
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

    # Filters header
    dbc.Row([
        dbc.Col([
            html.Label('Filters', className='font-weight-bold'),
        ], width=12)
    ]),

    # Filter dropdowns (generated dynamically)
    dbc.Row(id='filter-dropdowns'),

    # Heatmap value selection (only visible for heatmap plots)
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

    # Graph container
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='plot', style={'height': '600px', 'background-color': '#f8f9fa'}),
            width=12
        )
    ]),

    # Buttons row: "Mean over x-axis" and "Reset"
    dbc.Row([
        dbc.Col([
            dbc.Button("Mean over x-axis", id="group-button", color="primary", className="mr-2", n_clicks=0),
            dbc.Button("Reset", id="reset-button", color="secondary", className="mr-2", n_clicks=0)
        ], width=12, className="text-center mt-4")
    ])
], fluid=True, style={
    'padding': '40px',
    'backgroundColor': '#f8f9fa',
    'minHeight': '100vh'
})

# ---------------------------
# Callback: Update Filter Dropdowns Based on Dataset
# ---------------------------
@app.callback(
    Output('filter-dropdowns', 'children'),
    Input('dataset-selection', 'value')
)
def update_filter_dropdowns(dataset_selection):
    """
    Dynamically update filter dropdowns when a new dataset is selected.
    """
    df = df_dict[dataset_selection]
    return generate_filter_dropdowns(df)

# ---------------------------
# Callback: Update Axis Options Based on Dataset
# ---------------------------
@app.callback(
    [Output('x-axis', 'options'),
     Output('y-axis', 'options'),
     Output('x-axis', 'value'),
     Output('y-axis', 'value')],
    Input('dataset-selection', 'value')
)
def update_axis_options(dataset_selection):
    """
    Populate the x-axis and y-axis dropdown options based on the selected dataset.
    """
    df = df_dict[dataset_selection]
    options = [{'label': col, 'value': col} for col in df.columns if col != 'legend_text']
    return options, options, None, None

# ---------------------------
# Callback: Update Layout and Plot
# ---------------------------
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
def update_layout_and_plot(dataset_selection, plot_type, x_axis, y_axis, heatmap_value,
                           n_clicks_group, n_clicks_reset, filter_values,
                           x_axis_state, y_axis_state):
    """
    Updates the figure based on user input:
      - "Mean over x-axis" button: groups the data by x-axis and calculates mean of y-axis.
      - Axis labels remain the user's original choices, with optional ', grouped by x-axis'
        appended to the chart title if grouping is applied.
      - Rounds numeric values to 3 decimals on axes and color scales.
    """
    global created_columns
    df = df_dict[dataset_selection]

    # Store original axes for labeling
    orig_x_axis = x_axis
    orig_y_axis = y_axis

    # Build a dictionary of filter selections from the pattern-matching dropdowns
    filter_values_dict = {}
    # The pattern-matching input is the 8th in the list, so index = 7
    # We'll iterate over them safely:
    if len(callback_context.inputs_list) > 7:
        pattern_input_ids = callback_context.inputs_list[7]  # This is a list of IDs or strings
        for inp_id, val in zip(pattern_input_ids, filter_values):
            # Some of these might be strings (e.g., 'group-button.n_clicks'), so we check type
            if isinstance(inp_id, dict):
                # Retrieve the index from the pattern
                filter_index = inp_id.get('id', {}).get('index')
                if filter_index:
                    filter_values_dict[filter_index] = val

    # Apply the filter selections
    for col, filter_val in filter_values_dict.items():
        if filter_val:
            df = df[~df[col].isin(filter_val)]

    # If no data remains after filtering
    if df.empty:
        return ([], {'display': 'none'}, px.scatter(title="No Data Available"))

    triggered = callback_context.triggered[0]['prop_id'].split('.')[0]

    # Check if "Mean over x-axis" was clicked
    grouping_applied = False
    if triggered == 'group-button' and x_axis_state and y_axis_state:
        df, y_axis = group_by_x_axis(df, x_axis_state, y_axis_state)
        grouping_applied = True

    # If y_axis is object, sort by it in descending order (for consistent plotting)
    if y_axis and df[y_axis].dtype == 'object':
        df = df.sort_values(by=y_axis, ascending=False)

    heatmap_value_options = [{'label': col, 'value': col} for col in df.columns if col not in created_columns]
    heatmap_style = {'display': 'block'} if plot_type == 'heatmap' else {'display': 'none'}

    # Build a clean chart title
    title_str = f"{plot_type.capitalize()} of {orig_y_axis} vs {orig_x_axis}"
    if grouping_applied:
        title_str += f", grouped by {orig_x_axis}"

    # Generate the figure with clean axis labels
    if not x_axis or not y_axis:
        fig = px.scatter(title="Please select X-axis and Y-axis values")
    elif plot_type == 'scatter':
        fig = px.scatter(
            df, x=x_axis, y=y_axis,
            color='legend_text' if 'legend_text' in df.columns else None,
            title=title_str,
            template='plotly_white',
            labels={x_axis: orig_x_axis, y_axis: orig_y_axis}
        )
    elif plot_type == 'histogram':
        fig = px.histogram(
            df, x=x_axis, y=y_axis, color=x_axis, text_auto=True,
            title=title_str, template='plotly_white',
            histfunc='avg' if y_axis else 'count',
            labels={x_axis: orig_x_axis, y_axis: orig_y_axis}
        )
    elif plot_type == 'box':
        fig = px.box(
            df, x=x_axis, y=y_axis, color=x_axis,
            title=title_str, template='plotly_white',
            labels={x_axis: orig_x_axis, y_axis: orig_y_axis}
        )
    elif plot_type == 'strip':
        fig = px.strip(
            df, x=x_axis, y=y_axis,
            color='legend_text' if 'legend_text' in df.columns else None,
            title=title_str, template='plotly_white',
            labels={x_axis: orig_x_axis, y_axis: orig_y_axis}
        )
    elif plot_type == 'heatmap':
        if heatmap_value:
            try:
                agg_df = df.groupby([x_axis, y_axis], as_index=False).agg({heatmap_value: 'mean'})
                fig = px.density_heatmap(
                    agg_df,
                    x=x_axis,
                    y=y_axis,
                    z=heatmap_value,
                    color_continuous_scale='Viridis',
                    title=title_str,
                    template='plotly_white',
                    labels={x_axis: orig_x_axis, y_axis: orig_y_axis, heatmap_value: heatmap_value}
                )
            except Exception as e:
                print(f"Error during heatmap grouping: {e}")
                fig = px.scatter(title=f"Error generating heatmap: {e}")
        else:
            fig = px.scatter(title="Please select a heatmap value")
    else:
        fig = px.scatter(title="Invalid plot type selected")

    # Round numeric values on the axes and color scale to 3 decimals
    fig.update_layout(
        height=600,
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa',
        xaxis=dict(tickformat=".3f"),   # Round x-axis to 3 decimals
        yaxis=dict(tickformat=".3f")    # Round y-axis to 3 decimals
    )
    # For heatmap or other color scales, if present
    if 'coloraxis' in fig.layout:
        fig.update_layout(coloraxis_colorbar=dict(tickformat=".3f"))

    return heatmap_value_options, heatmap_style, fig

# ---------------------------
# Run the Dash Application
# ---------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
