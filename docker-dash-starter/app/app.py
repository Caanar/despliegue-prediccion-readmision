import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output
import requests
import json
from loguru import logger
import os


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app server
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# PREDICTION API URL 
api_url = os.getenv('API_URL')
api_url = "http://{}:8001/api/v1/predict".format(api_url)

# Layout in HTML
app.layout = html.Div(
    [
    html.H6("Ingrese la información del cliente:"),
    html.Div(["Número de procedimientos de laboratorio realizados en el encuentro: ",
              dcc.Input(id='num_lab_procedures', value='1', type='number')]),
    html.Br(),
    html.Div(["Número de medicamentos tomados por el cliente durante el encuentro: ",
              dcc.Input(id='num_medications', value='1', type='number')]),
    html.Br(),
    html.Div(["Número de veces que el paciente ha ingresado por urgencias en el año anterior al encuentro: ",
              dcc.Input(id='number_inpatient', value='1', type='number')]),
    html.Br(),
    html.Div(["Tiempo que el paciente ha pasado en el hospital (días): ",
              dcc.Input(id='time_in_hospital', value='1', type='number')]),
    html.Br(),
    html.Div(["ID que señala la manera en que el paciente fue dado de alta: ",
              dcc.Input(id='discharge_disposition_id', value='1', type='number')]),
    html.Br(),
    html.Div(["Número de diagnósticos que tiene el paciente en el sistema: ",
              dcc.Input(id='number_diagnoses', value='1', type='number')]),
    html.Br(),
    html.H6(html.Div(id='resultado')),
    ]
)

# Method to update prediction
@app.callback(
    Output(component_id='resultado', component_property='children'),
    [Input(component_id='num_lab_procedures', component_property='value'), 
     Input(component_id='num_medications', component_property='value'), 
     Input(component_id='number_inpatient', component_property='value'), 
     Input(component_id='time_in_hospital', component_property='value'),
     Input(component_id='discharge_disposition_id', component_property='value'),
     Input(component_id='number_diagnoses', component_property='value')]
)
def update_output_div(num_lab_procedures, num_medications, number_inpatient, time_in_hospital, discharge_disposition_id, number_diagnoses ):
    myreq = {
        "inputs": [
            {
            "num_lab_procedures": num_lab_procedures,
            "num_medications": num_medications,
            "number_inpatient": number_inpatient,
            "time_in_hospital": time_in_hospital,
            "discharge_disposition_id": discharge_disposition_id,
            "number_diagnoses": number_diagnoses
            }
        ]
      }
    headers =  {"Content-Type":"application/json", "accept": "application/json"}

    # POST call to the API
    response = requests.post(api_url, data=json.dumps(myreq), headers=headers)
    data = response.json()
    logger.info("Response: {}".format(data))

    # Pick result to return from json format
    prediction = data["predictions"][0]
    result = "<30" if prediction == 0 else ">30" if prediction == 1 else "NO"
    
    return result 

 

if __name__ == '__main__':
    logger.info("Running dash")
    app.run_server(debug=True)
