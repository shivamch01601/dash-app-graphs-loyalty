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
def home_page():
    return html.Div([
        html.H1("Airline Loyalty Dashboard"),
        dcc.Link("1. Earn vs Burn Over Time", href="/graph1"), html.Br(),
        dcc.Link("2. Earn by Channel", href="/earn"), html.Br(),
        dcc.Link("3. Burn by Channel", href="/burn"), html.Br(),
        dcc.Link("4. Monthly Earn", href="/earn-monthly"), html.Br(),
        dcc.Link("5. Monthly Burn", href="/burn-monthly"), html.Br(),
        dcc.Link("6. Earn/Burn Distribution", href="/distribution"), html.Br(),
        dcc.Link("7. Redemption Rate by Age", href="/age-rate"), html.Br(),
        dcc.Link("8. Redemption Rate by Channel", href="/channel-rate"), html.Br(),
        dcc.Link("9. Loyalty Program Modelling", href="/model"), html.Br(),
    ])

# ================= PAGE LAYOUTS =================
def graph1_page():
    return html.Div([
        html.H1("Historical Earn and Burn Over Time"),
        dcc.RangeSlider(id='year_range_slider_main', min=1997, max=2022, value=[1997,2022],
                        marks={y:str(y) for y in range(1997,2023)}),
        dcc.Graph(id='graph1-main')
    ])

def earn_page():
    return html.Div([
        html.H1("Historical Earn with Different Channels"),
        dcc.RangeSlider(id='year_range_slider_earn', min=1997, max=2022, value=[1997,2022],
                        marks={y:str(y) for y in range(1997,2023)}),
        dcc.Dropdown(id='channel_selection_earn',
                     options=[{'label':'Flight','value':'fl_earn'},
                              {'label':'Credit Card','value':'cc_earn'},
                              {'label':'Other','value':'ot_earn'}],
                     multi=True,
                     value=['fl_earn','cc_earn','ot_earn']),
        dcc.Graph(id='graph_earn_1')
    ])

def burn_page():
    return html.Div([
        html.H1("Historical Burn with Different Channels"),
        dcc.RangeSlider(id='year_range_slider_burn', min=1997, max=2022, value=[1997,2022],
                        marks={y:str(y) for y in range(1997,2023)}),
        dcc.Dropdown(id='channel_selection_burn',
                     options=[{'label':'Flight','value':'fl_burn'},
                              {'label':'Credit Card','value':'cc_burn'},
                              {'label':'Other','value':'ot_burn'}],
                     multi=True,
                     value=['fl_burn','cc_burn','ot_burn']),
        dcc.Graph(id='graph_burn_1')
    ])

def earn_monthly_page():
    return html.Div([
        html.H1("Historical Earn - Monthly"),
        dcc.RangeSlider(id='year_range_slider_earn_monthly', min=1997, max=2022, value=[1997,2022],
                        marks={y:str(y) for y in range(1997,2023)}),
        dcc.Dropdown(id='channel_selection_earn_monthly',
                     options=[{'label':'Flight','value':'fl_earn'},
                              {'label':'Credit Card','value':'cc_earn'},
                              {'label':'Other','value':'ot_earn'}],
                     multi=True,
                     value=['fl_earn','cc_earn','ot_earn']),
        dcc.Graph(id='graph_earn_monthly')
    ])

def burn_monthly_page():
    return html.Div([
        html.H1("Historical Burn - Monthly"),
        dcc.RangeSlider(id='year_range_slider_burn_monthly', min=1997, max=2022, value=[1997,2022],
                        marks={y:str(y) for y in range(1997,2023)}),
        dcc.Dropdown(id='channel_selection_burn_monthly',
                     options=[{'label':'Flight','value':'fl_burn'},
                              {'label':'Credit Card','value':'cc_burn'},
                              {'label':'Other','value':'ot_burn'}],
                     multi=True,
                     value=['fl_burn','cc_burn','ot_burn']),
        dcc.Graph(id='graph_burn_monthly')
    ])

def distribution_page():
    return html.Div([
        html.H1("Earn vs Burn Distribution"),
        dcc.Graph(id='earn-burn-histogram')
    ])

def age_rate_page():
    return html.Div([
        html.H1("Historical Redemption Rate by Age"),
        dcc.Graph(id='graph_age_range_vs_r_rate_1')
    ])

def channel_rate_page():
    return html.Div([
        html.H1("Redemption Rate by Channel"),
        dcc.Dropdown(id='channel_selection_rate',
                     options=[{'label':'Flight','value':'fl_r_rate'},
                              {'label':'Credit Card','value':'cc_r_rate'},
                              {'label':'Other','value':'ot_r_rate'}],
                     multi=True,
                     value=['fl_r_rate','cc_r_rate','ot_r_rate']),
        dcc.Graph(id='graph_age_range_vs_r_rate_channels')
    ])

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

@app.callback(Output('graph1-main','figure'), Input('year_range_slider_main','value'))
def graph1_cb(r):
    f = df1[(df1['year']>=r[0])&(df1['year']<=r[1])]
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Earn','Burn'))
    fig.add_trace(go.Scatter(x=f['year'], y=f['earn'], mode='lines+markers'), row=1,col=1)
    fig.add_trace(go.Scatter(x=f['year'], y=f['burn'], mode='lines+markers', line=dict(color='red')), row=2,col=1)
    fig.update_layout(height=800, showlegend=False)
    return fig

@app.callback(Output('graph_earn_1','figure'),
              Input('year_range_slider_earn','value'),
              Input('channel_selection_earn','value'))
def earn_cb(r,ch):
    f=df2[(df2['year']>=r[0])&(df2['year']<=r[1])]
    fig=go.Figure()
    for c in ch: fig.add_trace(go.Scatter(x=f['year'], y=f[c], mode='lines+markers', name=c))
    return fig

@app.callback(Output('graph_burn_1','figure'),
              Input('year_range_slider_burn','value'),
              Input('channel_selection_burn','value'))
def burn_cb(r,ch):
    f=df2[(df2['year']>=r[0])&(df2['year']<=r[1])]
    fig=go.Figure()
    for c in ch: fig.add_trace(go.Scatter(x=f['year'], y=f[c], mode='lines+markers', name=c))
    return fig

@app.callback(Output('graph_earn_monthly','figure'),
              Input('year_range_slider_earn_monthly','value'),
              Input('channel_selection_earn_monthly','value'))
def earn_m_cb(r,ch):
    f=df3[(df3['year']>=r[0])&(df3['year']<=r[1])]
    fig=go.Figure()
    for c in ch: fig.add_trace(go.Scatter(x=f['month_year'], y=f[c], mode='lines+markers', name=c))
    return fig

@app.callback(Output('graph_burn_monthly','figure'),
              Input('year_range_slider_burn_monthly','value'),
              Input('channel_selection_burn_monthly','value'))
def burn_m_cb(r,ch):
    f=df3[(df3['year']>=r[0])&(df3['year']<=r[1])]
    fig=go.Figure()
    for c in ch: fig.add_trace(go.Scatter(x=f['month_year'], y=f[c], mode='lines+markers', name=c))
    return fig

@app.callback(Output('earn-burn-histogram','figure'), Input('url','pathname'))
def dist_cb(_):
    fig=go.Figure()
    fig.add_bar(x=df4['earn_range'], y=df4['earn_count'], name='Earn')
    fig.add_bar(x=df4['burn_range'], y=df4['burn_count'], name='Burn')
    fig.update_layout(barmode='group')
    return fig

@app.callback(Output('graph_age_range_vs_r_rate_1','figure'), Input('url','pathname'))
def age_cb(_):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df5['age_range'], y=df5['r_rate'], mode='lines+markers'))
    return fig

@app.callback(Output('graph_age_range_vs_r_rate_channels','figure'),
              Input('channel_selection_rate','value'))
def channel_rate_cb(ch):
    fig=go.Figure()
    for c in ch: fig.add_trace(go.Scatter(x=df5['age_range'], y=df5[c], mode='lines+markers', name=c))
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
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=66, test_size=0.3)

    # Train Logistic Regression model
    log_reg = LogisticRegression()
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
    CVscore = cross_val_score(LogisticRegression(), x, y, cv=3, scoring='precision')

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
