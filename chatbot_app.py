import dash
from dash import Dash, callback, dcc, html
from dash.dependencies import Input, Output, State
import subprocess
from jupyter_dash import JupyterDash
import re
import dash_bootstrap_components as dbc


app = Dash(external_stylesheets=[dbc.themes.LUX])
app.layout = html.Div([
    html.H2('GNAR Chatbot Interface', style={'textAlign': 'center'}),
    html.H5('''This chatbot is built upon OpenAI LLM "gpt-4o-mini"
     and has been augmented to with academic papers relating to 
     Generalised Network Autoregressive models''', style={'textAlign': 'center'}),
    html.Br(),
    html.Label('Ask your question:', style = {'marginLeft': '4%' }),
    html.Br(),
    dcc.Textarea(id='question-area', value=None, style={'width': '50%', 'height': 50, 'marginLeft': '4%' }),
    html.Br(),
    html.Button(id='submit-btn', children='Submit', style= {'marginLeft': '4%' }),
    
    # Loading component to show spinner while fetching data
    dcc.Loading(id="loading-response", type="circle", children=[
        html.Br(),
        html.Div(id='response-area', children='', style= {'marginLeft': '4%' } ),
        html.Br(),
        html.Div(id='sources-area', children='', style= {'marginLeft': '4%' } ),
        html.Br(),
        html.Div(id='context-area', children='', style= {'marginLeft': '4%' } )

    ])
])

@callback(
    [Output('response-area', 'children'),
     Output('sources-area', 'children'),
     Output('context-area', 'children')],
    Input('submit-btn', 'n_clicks'),
    State('question-area', 'value'),
    prevent_initial_call=True
)
def create_response(_, question):
    
    if question is not None:

        # Generate response
        output = subprocess.run(['python', 'faiss_query.py', question], capture_output=True, text=True).stdout

        # Regex to capture context, response, and sources
        regex = r"Context:(.*?)Question:.*?Response:(.*?)Sources:(.*)"

        # Applying the regex
        matches = re.search(regex, output, re.DOTALL)
        
        if matches:
            context = matches.group(1).strip()
            response = matches.group(2).strip()
            sources = matches.group(3).strip()

            # Split the context into chunks based on '---' delimiter
            context_chunks = context.split('---')

            # Create a list format for context
            context_list = html.Div([
                html.Strong("Context:"),
                html.Ul([html.Li(chunk.strip()) for chunk in context_chunks if chunk.strip()])
            ])


        # Format the output for response and sources
        sources = html.Div([html.Strong("Sources:"),
                           html.Div(sources),
                           ])
        answer = html.Div([html.Strong("Answer:"),
                           html.Div(response),
                           ])
        
        
    return answer, sources, context_list


if __name__ == '__main__':
    app.run_server(debug=False)