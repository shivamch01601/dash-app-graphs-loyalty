# ================= LIBRARIES =================
import dash
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report

# ================= DATA =================
df = pd.read_csv("df_new.csv")
df1 = pd.read_csv("df1.csv", sep='|')
df2 = pd.read_csv("df2.csv", sep='|')
df3 = pd.read_csv("df3.csv", sep='|')
df4 = pd.read_csv("df4.csv", sep='|')
df5 = pd.read_csv("df5.csv", sep='|')

# ================= STYLES =================
table_style = {
    'borderCollapse': 'collapse',
    'margin': '10px auto',
    'textAlign': 'center',
    'fontFamily': 'Arial',
    'fontSize': '14px',
    'width': '50%'
}

# ================= APP =================
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Location(id='url'),
    html.Div(id='page-content', style={'width': '90%', 'margin': '0 auto', 'textAlign': 'center'})
])

# ================= HOME PAGE =================
# ================= HOME PAGE =================
def home_page():
    return html.Div([
        # Main Heading
        html.H1(
            "Welcome to our Interactive Dashboard for Airline Loyalty Program Data!",
            style={
                'color': '#0d47a1', 
                'textAlign': 'center', 
                'fontSize': '32px', 
                'fontWeight': 'bold',
                'marginBottom': '20px'
            }
        ),
        
        # Subheading / Description
        html.P(
            "This powerful tool is designed to help you explore, visualize, and analyze "
            "the earning and redemption activities of a hypothetical airline's loyalty program customers. "
            "By leveraging this dashboard, you can gain valuable insights into customer behavior, preferences, "
            "and trends, which enable businesses to make data-driven decisions to enhance the loyalty program.",
            style={
                'color': '#424242', 
                'textAlign': 'center', 
                'fontSize': '18px', 
                'margin': '0 auto 30px auto', 
                'width': '80%', 
                'lineHeight': '1.6'
            }
        ),
        
        # Call to action
        html.P(
            "Please click on the section below that you want to explore:",
            style={
                'color': '#f57c00', 
                'textAlign': 'center', 
                'fontSize': '20px', 
                'fontWeight': 'bold',
                'marginBottom': '30px'
            }
        ),
        
        # Links to pages
        html.Div([
            dcc.Link("1. Earn vs Burn Over Time", href="/graph1", style={'display': 'block', 'margin': '10px', 'fontSize': '18px'}),
            dcc.Link("2. Earn by Channel", href="/earn", style={'display': 'block', 'margin': '10px', 'fontSize': '18px'}),
            dcc.Link("3. Burn by Channel", href="/burn", style={'display': 'block', 'margin': '10px', 'fontSize': '18px'}),
            dcc.Link("4. Monthly Earn", href="/earn-monthly", style={'display': 'block', 'margin': '10px', 'fontSize': '18px'}),
            dcc.Link("5. Monthly Burn", href="/burn-monthly", style={'display': 'block', 'margin': '10px', 'fontSize': '18px'}),
            dcc.Link("6. Earn/Burn Distribution", href="/distribution", style={'display': 'block', 'margin': '10px', 'fontSize': '18px'}),
            dcc.Link("7. Redemption Rate by Age", href="/age-rate", style={'display': 'block', 'margin': '10px', 'fontSize': '18px'}),
            dcc.Link("8. Redemption Rate by Channel", href="/channel-rate", style={'display': 'block', 'margin': '10px', 'fontSize': '18px'}),
            dcc.Link("9. Loyalty Program Modelling", href="/model", style={'display': 'block', 'margin': '10px', 'fontSize': '18px'}),
        ], style={'width': '50%', 'margin': '0 auto', 'textAlign': 'center'})
    ], style={'padding': '50px 20px', 'backgroundColor': '#e3f2fd'})


# ================= PAGE LAYOUTS =================
def graph1_page():
    return html.Div(
        style={'textAlign': 'center', 'width': '80%', 'margin': 'auto'},
        children=[
            # Layout for graph 1
            html.Div([
                html.Div([
                    html.H3("", style={'color': 'blue'}),
                    html.P("", style={'color': 'red', 'font-weight': 'bold'}),
                ], style={'color': 'blue', 'font-weight': 'bold'}),
                
                html.H1("Historical Earn and Burn Over Time"),
                
                # Text description for Graph 1
                html.Div([
                    html.P(
                        "This graph shows the historical earn and burn over time. "
                        "You can adjust the year range using the slider below."
                    )
                ], style={'margin-bottom': '20px'}),
                
                html.Div([
                    html.H3("Select Year Range"),
                    dcc.RangeSlider(
                        id='year_range_slider_main',
                        min=1997,
                        max=2022,
                        value=[1997, 2022],
                        marks={year: str(year) for year in range(1997, 2023)},
                        step=1
                    )
                ], style={'margin-bottom': '20px'}),
                
                # Graph with description text at the top
                dcc.Graph(
                    id='graph1-main',
                    style={'width': '100%', 'float': 'left'}
                ),
            ])
        ]
    )


def earn_page():
    return html.Div([
        html.H1("Historical Earn with Different Channels"),

        # Text description for Graph 2
        html.Div([
            html.P(
                "This graph shows the historical earn with different channels over time. "
                "You can adjust the year range and select channels using the sliders and dropdowns below."
            )
        ], style={'margin-bottom': '20px'}),

        html.H3("Select Year Range"),
        dcc.RangeSlider(
            id='year_range_slider_earn',
            min=1997,
            max=2022,
            value=[1997, 2022],
            marks={year: str(year) for year in range(1997, 2023)},
            step=1
        ),
        html.H3("Select Channels to Display"),
        dcc.Dropdown(
            id='channel_selection_earn',
            options=[
                {'label': 'Miles Earned by Flight Activity (fl)', 'value': 'fl_earn'},
                {'label': 'Miles Earned through Credit Card (cc)', 'value': 'cc_earn'},
                {'label': 'Miles Earned through Other Channels (ot)', 'value': 'ot_earn'}
            ],
            multi=True,
            value=['fl_earn', 'cc_earn', 'ot_earn']
        ),
        dcc.Graph(id='graph_earn_1')
    ], style={'margin-bottom': '20px'})

def burn_page():
    return html.Div([
        html.H1("Historical Burn with Different Channels"),

        # Text description for Graph 2 - Burn
        html.Div([
            html.P(
                "This graph shows the historical burn with different channels over time. "
                "You can adjust the year range and select channels using the sliders and dropdowns below."
            )
        ], style={'margin-bottom': '20px'}),

        html.H3("Select Year Range"),
        dcc.RangeSlider(
            id='year_range_slider_burn',
            min=1997,
            max=2022,
            value=[1997, 2022],
            marks={year: str(year) for year in range(1997, 2023)},
            step=1
        ),
        html.H3("Select Channels to Display"),
        dcc.Dropdown(
            id='channel_selection_burn',
            options=[
                {'label': 'Miles Burned by Flight Activity (fl)', 'value': 'fl_burn'},
                {'label': 'Miles Burned through Credit Card (cc)', 'value': 'cc_burn'},
                {'label': 'Miles Burned through Other Channels (ot)', 'value': 'ot_burn'}
            ],
            multi=True,
            value=['fl_burn', 'cc_burn', 'ot_burn']
        ),
        dcc.Graph(id='graph_burn_1')
    ], style={'margin-bottom': '20px'})
    


def earn_monthly_page():
    return html.Div([
        html.H1("Historical Earn with Different Channels - Monthly"),

        # Text description for Graph 3-a
        html.Div([
            html.P(
                "This graph shows the historical earn with different channels on a monthly basis. "
                "You can adjust the year range and select channels using the sliders and dropdowns below."
            )
        ], style={'margin-bottom': '20px'}),

        html.H3("Select Year Range"),
        dcc.RangeSlider(
            id='year_range_slider_earn_monthly',
            min=1997,
            max=2022,
            value=[1997, 2022],
            marks={year: str(year) for year in range(1997, 2023)},
            step=1
        ),
        html.H3("Select Channels to Display"),
        dcc.Dropdown(
            id='channel_selection_earn_monthly',
            options=[
                {'label': 'Miles Earned by Flight Activity (fl)', 'value': 'fl_earn'},
                {'label': 'Miles Earned through Credit Card (cc)', 'value': 'cc_earn'},
                {'label': 'Miles Earned through Other Channels (ot)', 'value': 'ot_earn'}
            ],
            multi=True,
            value=['fl_earn', 'cc_earn', 'ot_earn']
        ),
        dcc.Graph(id='graph_earn_monthly')
    ], style={'margin-bottom': '20px'})

def burn_monthly_page():
    return html.Div([
        html.H1("Historical Burn with Different Channels - Monthly"),

        # Text description for Graph 3-b
        html.Div([
            html.P(
                "This graph shows the historical burn with different channels on a monthly basis. "
                "You can adjust the year range and select channels using the sliders and dropdowns below."
            )
        ], style={'margin-bottom': '20px'}),

        html.H3("Select Year Range"),
        dcc.RangeSlider(
            id='year_range_slider_burn_monthly',
            min=1997,
            max=2022,
            value=[1997, 2022],
            marks={year: str(year) for year in range(1997, 2023)},
            step=1
        ),
        html.H3("Select Channels to Display"),
        dcc.Dropdown(
            id='channel_selection_burn_monthly',
            options=[
                {'label': 'Miles Burned by Flight Activity (fl)', 'value': 'fl_burn'},
                {'label': 'Miles Burned through Credit Card (cc)', 'value': 'cc_burn'},
                {'label': 'Miles Burned through Other Channels (ot)', 'value': 'ot_burn'}
            ],
            multi=True,
            value=['fl_burn', 'cc_burn', 'ot_burn']
        ),
        dcc.Graph(id='graph_burn_monthly')
    ], style={'margin-bottom': '20px'})

def distribution_page():
    return html.Div([
        html.H1("Historical Earn-Burn Distribution"),

        # Text description for Graph 4 and 5
        html.Div([
            html.P(
                "This graph shows the distribution of historical earn and burn. "
                "You can check the distribution of 'earn' and 'burn' separately by clicking on the checkbox."
            )
        ], style={'margin-bottom': '20px'}),

        dcc.Graph(id='earn-burn-histogram'),
        html.Button(id='dummy-input', style={'display': 'none'})  # Dummy input (hidden button)
    ])

def age_rate_page():
    return html.Div([
        html.H1("Historical Redemption Rate"),

        # Text description for Graph 6
        html.Div([
            html.P(
                "This graph shows the historical redemption rate by age range. "
                "Each point on the graph represents a specific age range and its corresponding redemption rate."
            )
        ], style={'margin-bottom': '20px'}),

        dcc.Graph(id='graph_age_range_vs_r_rate_1')
    ], style={'margin-bottom': '20px'})

def channel_rate_page():
    return html.Div([
        html.H1("Historical Redemption Rate for Channels"),

        # Text description for Graph 7
        html.Div([
            html.P(
                "This graph shows the historical redemption rate for different channels by age range. "
                "You can select channels using the dropdown below."
            )
        ], style={'margin-bottom': '20px'}),

        html.H3("Select Channels to Display"),
        dcc.Dropdown(
            id='channel_selection_rate',
            options=[
                {'label': 'Flight Redemption Rate (fl)', 'value': 'fl_r_rate'},
                {'label': 'Credi-Card Redemption Rate (cc)', 'value': 'cc_r_rate'},
                {'label': 'Redemption Rate for Other Channels (ot)', 'value': 'ot_r_rate'},
            ],
            multi=True,
            value=['fl_r_rate', 'cc_r_rate', 'ot_r_rate'],
        ),
        dcc.Graph(id='graph_age_range_vs_r_rate_channels')
    ], style={'margin-bottom': '20px'})

def model_page():
    return html.Div([
        html.H1("Loyalty Program Modelling Exercise"),
        html.Div([
            f"In this predictive model exercise, our objective is to predict the status of each airline user based on selected features. The target variable can be chosen from three options: whether the user is an active member of airline's loyalty program, whether he or she uses an airline co-branded credit card, or whether he or she is an active redeemer of miles earned through airline's loyalty program participation. A value of 1 signifies active status (0 for inactive), credit card usage (0 for non-credit card user), or redemption of miles (0 for non-redemption).",
            html.P("This section allows you to evaluate a logistic regression model on the given dataset. Select features, target variable, and threshold. Click 'Run Evaluation' to view the results."),
        ], style={'marginBottom': '20px'}),
        html.Label("Select Features", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='feature-selector',
            options=[{'label': col, 'value': col} for col in df.columns],
            multi=True,
            value=df.columns[2:].tolist()
        ),
        html.Br(),
        html.Label("Select Target Variable", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='target-variable',
            options=[{'label': col, 'value': col} for col in df.columns[1:4]],
            value=df.columns[1]
        ),
        html.Br(),
        html.Label("Select Threshold (Probability Cutoff)", style={'fontWeight': 'bold'}),
        html.P("The threshold determines the probability cutoff for classifying instances. Changing it alters how predictions are made based on probabilities."),
        dcc.Slider(
            id='threshold-slider',
            min=0.1,
            max=1.0,
            step=0.1,
            value=0.5,
            marks={i / 10: str(i / 10) for i in range(1, 11)}
        ),
        html.Br(),
        html.Div([
            "Once you click 'Run Evaluation', please wait for 5 to 10 seconds for model training."
        ], style={'marginTop': '20px'}),
        html.Br(),
        html.Button('Run Evaluation', id='run-evaluation'),
        html.Br(),
        html.Hr(style={'border-top': '2px solid black', 'font-weight': 'bold'}),
        html.Div(id='evaluation-output')
    ])

# ================= ROUTER =================
@app.callback(Output('page-content','children'), Input('url','pathname'))
def route(path):
    pages = {
        '/graph1': graph1_page,
        '/earn': earn_page,
        '/burn': burn_page,
        '/earn-monthly': earn_monthly_page,
        '/burn-monthly': burn_monthly_page,
        '/distribution': distribution_page,
        '/age-rate': age_rate_page,
        '/channel-rate': channel_rate_page,
        '/model': model_page,
    }
    return pages.get(path, home_page)()

# ================= CALLBACKS =================

@app.callback(
    Output('graph1-main', 'figure'),
    Input('year_range_slider_main', 'value')
)
def update_graph1_main(year_range):
    filtered_data = df1[(df1['year'] >= year_range[0]) & (df1['year'] <= year_range[1])]

    # Create subplot for 'earn' and 'burn'
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Earn Over Time', 'Burn Over Time'))

    # Subplot for 'earn'
    trace_earn = go.Scatter(x=filtered_data['year'], y=filtered_data['earn'], mode='lines+markers', name='Earn')
    fig.add_trace(trace_earn, row=1, col=1)

    # Subplot for 'burn'
    trace_burn = go.Scatter(x=filtered_data['year'], y=filtered_data['burn'], mode='lines+markers', name='Burn', line=dict(color='red'))
    fig.add_trace(trace_burn, row=2, col=1)

    # Update subplot layout
    fig.update_layout(height=800, showlegend=False)

    # Update x-axis and y-axis labels
    fig.update_xaxes(title_text='Year', row=2, col=1)
    fig.update_yaxes(title_text='Miles ', row=1, col=1)
    fig.update_yaxes(title_text='Miles ', row=2, col=1)

    return fig


# Callback for plot_earn -- graph 2
@app.callback(
    Output('graph_earn_1', 'figure'),
    Input('year_range_slider_earn', 'value'),
    Input('channel_selection_earn', 'value')
)
def update_graph_earn(year_range, selected_channels):
    filtered_data = df2[(df2['year'] >= year_range[0]) & (df2['year'] <= year_range[1])]

    # Create a Plotly figure
    fig = go.Figure()

    # Add line traces for each selected channel
    for channel in selected_channels:
        fig.add_trace(go.Scatter(
            x=filtered_data['year'],
            y=filtered_data[channel],
            mode='lines+markers',
            name=channel
        ))

    # Update layout
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Miles ',
        legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
        margin=dict(l=0, r=0, t=0, b=0)  # Adjust margins as needed
    )

    return fig



# Callback for plot_burn
@app.callback(
    Output('graph_burn_1', 'figure'),
    Input('year_range_slider_burn', 'value'),
    Input('channel_selection_burn', 'value')
)
def update_graph_burn(year_range, selected_channels):
    filtered_data = df2[(df2['year'] >= year_range[0]) & (df2['year'] <= year_range[1])]

    # Create a Plotly figure
    fig = go.Figure()

    # Add line traces for 'fl_burn'
    for channel in selected_channels[0:1]:
        fig.add_trace(go.Scatter(
            x=filtered_data['year'],
            y=filtered_data[channel],
            mode='lines+markers',
            name=channel,
            line=dict(color='green', width=2)
        ))

    # Add line traces for 'cc_burn' and 'ot_burn'
    for channel in selected_channels[1:]:
        fig.add_trace(go.Scatter(
            x=filtered_data['year'],
            y=filtered_data[channel],
            mode='lines+markers',
            name=channel,
            line=dict(width=2)
        ))

    # Update layout
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Miles',
        legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
        margin=dict(l=0, r=0, t=0, b=0)  # Adjust margins as needed
    )

    return fig

# Callback for plot_earn_monthly Graph 3-a
@app.callback(
    Output('graph_earn_monthly', 'figure'),
    Input('year_range_slider_earn_monthly', 'value'),
    Input('channel_selection_earn_monthly', 'value')
)
def update_graph_earn_monthly(year_range, selected_channels):
    filtered_data = df3[(df3['year'] >= year_range[0]) & (df3['year'] <= year_range[1])]

    # Create a Plotly figure
    fig = go.Figure()

    # Add line traces for 'earn'
    for channel in selected_channels:
        fig.add_trace(go.Scatter(
            x=filtered_data['month_year'],
            y=filtered_data[f'{channel.lower().replace(" ", "_")}'],
            mode='lines+markers',
            name=channel,
            marker=dict(symbol='circle', size=8),
            line=dict(width=2)
        ))

    # Update layout
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Miles',
        legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
        margin=dict(l=0, r=0, t=0, b=0)  # Adjust margins as needed
    )

    return fig

# Callback for plot_burn_monthly Graph 3-b
@app.callback(
    Output('graph_burn_monthly', 'figure'),
    Input('year_range_slider_burn_monthly', 'value'),
    Input('channel_selection_burn_monthly', 'value')
)
def update_graph_earn_monthly(year_range, selected_channels):
    filtered_data = df3[(df3['year'] >= year_range[0]) & (df3['year'] <= year_range[1])]

    # Create a Plotly figure
    fig = go.Figure()

    # Add line traces for 'earn'
    for channel in selected_channels:
        fig.add_trace(go.Scatter(
            x=filtered_data['month_year'],
            y=filtered_data[f'{channel.lower().replace(" ", "_")}'],
            mode='lines+markers',
            name=channel,
            marker=dict(symbol='circle', size=8),
            line=dict(width=2)
        ))

    # Update layout
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Miles',
        legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
        margin=dict(l=0, r=0, t=0, b=0)  # Adjust margins as needed
    )

    return fig

    # Callback for graph 4 and 5
@app.callback(
    Output('earn-burn-histogram', 'figure'),
    [Input('dummy-input', 'n_clicks')]  # Dummy input to trigger the callback
)
def update_graph(n_clicks):
    # Create the Plotly figure
    fig = go.Figure()

    # Adding histogram trace for 'earn_count'
    fig.add_trace(go.Bar(x=df4['earn_range'], y=df4['earn_count'], name='Earn'))

    # Adding histogram trace for 'burn_count'
    fig.add_trace(go.Bar(x=df4['burn_range'], y=df4['burn_count'], name='Burn'))

    # Updating layout
    fig.update_layout(barmode='group', xaxis_title='Ranges', yaxis_title='Count')

    return fig


# Callback for graph 6 
# ================= FIXED CALLBACK FOR GRAPH 6 (NO LAYOUT CHANGE NEEDED) =================

@app.callback(
    Output('graph_age_range_vs_r_rate_1', 'figure'),
    Input('url', 'pathname'),  # ← Uses existing global url (always available)
    prevent_initial_call=False
)
def update_graph_age_range_vs_r_rate(pathname):
    # Only render on age-rate page
    if '/age-rate' not in pathname:
        return go.Figure()  # Empty on other pages
    
    sns.set_palette("husl")

    age_order = ['0-1', '1-5', '5-10', '10-20', '20-30', '30-40', '40-50+']
    df5['age_range'] = pd.Categorical(df5['age_range'], categories=age_order, ordered=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df5['age_range'],
        y=df5['r_rate'],
        mode='lines+markers',
        marker=dict(symbol='circle', size=8),
        line=dict(color='blue', width=2),
        name='R Rate'
    ))

    fig.update_layout(
        xaxis_title='Age Range',
        yaxis_title='Redemption Rate',
        font=dict(size=15),
        showlegend=False
    )

    for i, txt in enumerate(df5['r_rate']):
        fig.add_annotation(
            x=df5['age_range'].iloc[i],
            y=df5['r_rate'].iloc[i],
            text=f'{txt:.2f}',
            bgcolor='lightblue',
            borderpad=0,
            showarrow=False,
            font=dict(size=10)
        )

    return fig

# Callback for graph 7
@app.callback(
    Output('graph_age_range_vs_r_rate_channels', 'figure'),
    [Input('channel_selection_rate', 'value')]
)
def update_graph_age_range_vs_r_rate_channels(selected_channels):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    for channel in selected_channels:
        fig.add_trace(go.Scatter(
            x=df5['age_range'],
            y=df5[channel],
            mode='lines+markers',
            marker=dict(symbol='circle', size=8),
            line=dict(width=2),
            name=f'{channel.split("_")[0].upper()} R Rate'
        ))

    fig.update_layout(
        xaxis_title='Age Range',
        yaxis_title='Redemption Rate',
        font=dict(size=15),
        showlegend=True
    )

    fig.update_xaxes(tickvals=['0-1', '1-5', '5-10', '10-20', '20-30', '30-40'], ticktext=['0-1', '1-5', '5-10', '10-20', '20-30', '30-40']
                    )

    for channel in selected_channels:
        for i, txt in enumerate(df5[channel]):
            fig.add_annotation(
                x=df5['age_range'][i],
                y=df5[channel][i],
                text=f'{channel.split("_")[0].upper()} {txt:.2f}',
                showarrow=False,
                font=dict(size=10),
                bgcolor='lightblue' if channel == 'fl_r_rate' else 'lightblue' if channel == 'cc_r_rate' else 'lightblue',
                borderpad=0  # Set borderpad to 0 to remove the box around the text
            )

    return fig

# Callback for model evaluation
@app.callback(
    Output('evaluation-output', 'children'),
    [Input('run-evaluation', 'n_clicks')],
    [State('feature-selector', 'value'),
     State('target-variable', 'value'),
     State('threshold-slider', 'value')]
)
def run_evaluation(n_clicks, selected_features, target_variable, threshold):
    if n_clicks is not None and n_clicks > 0:
        result = logistic_regression_evaluation(df, selected_features, target_variable, threshold)
        return result


def logistic_regression_evaluation(df, features, target, threshold=0.5):
    # Select features and target variable
    x = df[features]
    y = df[target]

    # Split the data into training and testing sets
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=66, test_size=0.3)

    # Train Logistic Regression model
    # old - log_reg = LogisticRegression()
    log_reg = LogisticRegression(max_iter=100, solver='liblinear', random_state=42)
    log_reg.fit(X_train, Y_train)

    # Make predictions on the training set
    y_train_pred = log_reg.predict_proba(X_train)[:, 1] >= threshold

    # Make predictions on the test set
    y_test_pred = log_reg.predict_proba(X_test)[:, 1] >= threshold

    # Calculate metrics
    train_accuracy = accuracy_score(Y_train, y_train_pred) * 100
    train_precision = precision_score(Y_train, y_train_pred) * 100
    test_accuracy = accuracy_score(Y_test, y_test_pred) * 100
    test_precision = precision_score(Y_test, y_test_pred) * 100
    cm = confusion_matrix(Y_test, y_test_pred)

    # Calculate confusion matrix percentages
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Format confusion matrix with color
    cm_html = html.Table(style=table_style, children=[
        html.Tr([
            html.Th('(Actual ↓) : (Predicted →)', style={'border': '1px solid black', 'padding': '5px'}),
            html.Th('Positive', style={'border': '1px solid black', 'padding': '5px'}),
            html.Th('Negative', style={'border': '1px solid black', 'padding': '5px'})
        ]),
        html.Tr([
            html.Th('Positive', style={'border': '1px solid black', 'padding': '5px'}),
            html.Td(f'{cm_perc[0][0]:.2f}%', style={'border': '1px solid black', 'padding': '5px',
                                                   'background-color': '#c8e6c9' if cm_perc[0][0] >= 50 else '#ffccbc'}),
            html.Td(f'{cm_perc[0][1]:.2f}%', style={'border': '1px solid black', 'padding': '5px',
                                                   'background-color': '#ffccbc' if cm_perc[0][1] >= 50 else '#c8e6c9'})
        ]),
        html.Tr([
            html.Th('Negative', style={'border': '1px solid black', 'padding': '5px'}),
            html.Td(f'{cm_perc[1][0]:.2f}%', style={'border': '1px solid black', 'padding': '5px',
                                                   'background-color': '#ffccbc' if cm_perc[1][0] >= 50 else '#c8e6c9'}),
            html.Td(f'{cm_perc[1][1]:.2f}%', style={'border': '1px solid black', 'padding': '5px',
                                                   'background-color': '#c8e6c9' if cm_perc[1][1] >= 50 else '#ffccbc'})
        ])
    ])

    # Perform cross-validation
    # old - CVscore = cross_val_score(LogisticRegression(), x, y, cv=10, scoring='precision')
    CVscore = cross_val_score(
    LogisticRegression(max_iter=200, solver='liblinear'),
    x, y,
    cv=3,              # was 10
    scoring='precision',
    n_jobs=1           # IMPORTANT: prevent parallel memory spikes
    )


    # Classification report
    class_report = classification_report(Y_test, y_test_pred, output_dict=True)

    # Prepare classification report for display
    class_report_rows = [
        html.Tr([
            html.Th("Class"),
            html.Th("Precision"),
            html.Th("Recall"),
            html.Th("F1-Score"),
            html.Th("Support")
        ]),
        html.Tr([
            html.Td("0"),
            html.Td(f'{class_report["0"]["precision"]:.2f}', style={'color': 'blue'}),
            html.Td(f'{class_report["0"]["recall"]:.2f}', style={'color': 'blue'}),
            html.Td(f'{class_report["0"]["f1-score"]:.2f}', style={'color': 'blue'}),
            html.Td(class_report["0"]["support"])
        ]),
        html.Tr([
            html.Td("1"),
            html.Td(f'{class_report["1"]["precision"]:.2f}', style={'color': 'blue'}),
            html.Td(f'{class_report["1"]["recall"]:.2f}', style={'color': 'blue'}),
            html.Td(f'{class_report["1"]["f1-score"]:.2f}', style={'color': 'blue'}),
            html.Td(class_report["1"]["support"])
        ])
    ]

    class_report_html = html.Table(style=table_style, children=class_report_rows)
    
    actual_train_count = Y_train.sum()
    predicted_train_count = y_train_pred.sum()

    # Test Set
    actual_test_count = Y_test.sum()
    predicted_test_count = y_test_pred.sum()

    bar_data = [
        go.Bar(
            x=['Training Set', 'Test Set'],
            y=[actual_train_count, actual_test_count],
            name='Actual',
            marker=dict(color='lightblue')
        ),
        go.Bar(
            x=['Training Set', 'Test Set'],
            y=[predicted_train_count, predicted_test_count],
            name='Predicted',
            marker=dict(color='lightgreen')
        )
    ]
    
    # Layout adjustments
    bar_layout = go.Layout(
    		title=dict(text='Actual vs Predicted Values Count', x=0.5, font=dict(size=20, color='black', family='Arial, sans-serif')),
    		xaxis=dict(title=dict(text='Dataset', font=dict(size=20, color='black', family='Arial, sans-serif'))),
    		yaxis=dict(title=dict(text='Count', font=dict(size=20, color='black', family='Arial, sans-serif'))),
    		legend=dict(font=dict(size=17, family='Arial, sans-serif')),
    		barmode='group',
    		bargap=0.2,
    		bargroupgap=0.1 )

    
    # Create figure
    bar_fig = go.Figure(data=bar_data, layout=bar_layout)

    # Return formatted results
    result = html.Div([
        html.Div([         
            html.Br(),
            html.B("Training Set Metrics:"),
            html.Br(),
            f'Percentage of correctly predicted instances (both positive and negative) out of all predictions made by the model [Accuracy] : ',
            html.Span(f'{train_accuracy:.2f}%', style={'color': 'blue'}),
            html.Br(),
            f'Percentage of correctly predicted positive instances out of all positive predictions [Precision] : ',
            html.Span(f'{train_precision:.2f}%', style={'color': 'blue'}),  # Highlight in blue
            html.Hr(), 
            html.Br()
        ]),
        html.Div([
            html.B("Test Set Metrics:"),
            html.Br(),
            f'Percentage of correctly predicted instances (both positive and negative) out of all predictions made by the model [Accuracy] : ',
            html.Span(f'{test_accuracy:.2f}%', style={'color': 'green'}),  # Highlight in green
            html.Br(),
            f'Percentage of correctly predicted positive instances out of all positive predictions [Precision] : ',
            html.Span(f'{test_precision:.2f}%', style={'color': 'green'}),  # Highlight in green
            html.Hr(), 
            html.Br()
        ]),
        html.Div([
            html.B("Confusion Matrix:"),
            cm_html,
            html.Br()
        ]),
        
        html.Hr(), 
        html.Br(),
        dcc.Graph(id='actual-vs-predicted-bar', figure=bar_fig),
        html.Hr(), 
        html.Br(),
        
        html.Div([
            html.B("Classification Report:"),
            class_report_html,  # Display classification report in table format
            html.Hr(), 
            html.Br()
        ]),
        html.Div([
            html.B("Actual and Predicted Values Count (Training):"),
            html.Br(),
            f'Actual 1 count - The number of instances in the dataset that belong to class 1 : ',
            html.Span(f'{Y_train.sum()}', style={'color': 'blue'}),
            html.Br(),
            f'Prediction 1 count - The number of instances predicted by the model to belong to class 1 : ',
            html.Span(f'{y_train_pred.sum()}', style={'color': 'blue'}),
            html.Hr(), 
            html.Br()
        ]),
        html.Div([
            html.B("Actual and Predicted Values Count (Test):"),
            html.Br(),
            f'Actual 1 count - The number of instances in the dataset that belong to class 1 : ',
            html.Span(f'{Y_test.sum()}', style={'color': 'green'}),
            html.Br(),
            f'Prediction 1 count - The number of instances predicted by the model to belong to class 1 : ',
            html.Span(f'{y_test_pred.sum()}', style={'color': 'green'}),
            html.Hr(), 
            html.Br()
        ]),
        html.Div([
            html.B("Cross Validation Score for Logistic Regression:"),
            html.Br(),
            f'Percentage of correctly predicted positive instances out of all positive predictions made by the model during cross-validation [Precision CV] : ',
            html.Span(f'{CVscore.mean() * 100:.2f}% (Mean)', style={'color': 'green'}),
            html.Br(),
            f'Standard deviation during cross-validation of precision measures the variability in positive prediction accuracy across different cross-validation folds : ',
            html.Span(f' {CVscore.std()}', style={'color': 'green'}),
            html.Hr(),
            html.Br(),
            
            
        ])
    ])

            
    return result

# ================= RUN =================
if __name__ == '__main__':
    app.run(debug=True)
