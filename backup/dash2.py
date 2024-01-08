import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime
import dash_bootstrap_components as dbc
import dash_table
from dash_table.Format import Format, Scheme
import pmdarima as pm

app = dash.Dash(__name__)

# Importar el DataFrame
df = pd.read_csv('./datos/archivofinal.csv')
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Crear una columna llamada Numero de dias
df['Numero de dias'] = (df['Fecha'] - df['Fecha'].min()).dt.days + 1

# Lista de columnas que se utilizarán en el modelo
feature_cols = ['Incremento', 'FCA', 'IPS', 'Sobrevivencia', 'Biomasa', 'Biomasa_Hectarea', 'Kg/Has']

# Crear un DataFrame para almacenar las predicciones
future_df = pd.DataFrame()

# Generar fechas futuras
last_date = df['Fecha'].max()
future_dates = pd.date_range(start=last_date, periods=70).tolist()[1:]
future_df['Fecha'] = future_dates
future_df['Numero de dias'] = (future_df['Fecha'] - df['Fecha'].min()).dt.days


# Estimar los valores de las características para las fechas futuras
for feature in feature_cols:
    # Crear un nuevo DataFrame que excluya las filas con valores NaN en la característica actual
    feature_df = df[['Numero de dias', feature]].dropna()

    # Entrenar un modelo de regresión lineal con los datos existentes
    model = LinearRegression()
    model.fit(feature_df[['Numero de dias']], feature_df[feature])

    # Utilizar el modelo para predecir los valores de la característica para las fechas futuras
    future_df[feature] = model.predict(future_df[['Numero de dias']])


# Combinar el DataFrame original con el DataFrame de predicciones
extended_df = pd.concat([df, future_df], ignore_index=True)


# Seleccionar las características y el objetivo para el modelo
features = extended_df[['Numero de dias'] + feature_cols]
target = extended_df['Peso_Promedio']

# Eliminar las filas con valores NaN en las características y el objetivo
valid_indices = features.dropna().index
features = features.loc[valid_indices]
target = target.loc[valid_indices]

# Entrenar el modelo final y realizar las predicciones
final_model = LinearRegression()
final_model.fit(features[:len(df)], df['Peso_Promedio'].dropna())
predictions = final_model.predict(features)

# Añadir las predicciones al DataFrame
extended_df.loc[valid_indices, 'Peso_Promedio'] = predictions

# Guardar el último valor real y la última fecha
last_real_value = df['Peso_Promedio'].iloc[-1]
last_real_date = df['Fecha'].iloc[-1]

# Reemplazar la primera predicción con el último valor real
extended_df.loc[len(df), 'Peso_Promedio'] = last_real_value

# Adicionalmente, asegúrate de que la primera fecha de predicción sea la misma que la última fecha real
extended_df.loc[len(df), 'Fecha'] = last_real_date

# Calcular las correlaciones
correlations = df[feature_cols + ['Peso_Promedio']].corr()['Peso_Promedio'].drop('Peso_Promedio')

# Seleccionar la característica con la mayor correlación absoluta
max_corr_feature = correlations.abs().idxmax()

# Entrenar un modelo de regresión lineal simple con la característica seleccionada
simple_model = LinearRegression()
simple_model.fit(df[[max_corr_feature]], df['Peso_Promedio'].dropna())

# Realizar las predicciones
extended_df['Predicciones_Simple'] = simple_model.predict(extended_df[[max_corr_feature]])

# Reemplazar la primera predicción simple con el último valor real
extended_df.loc[len(df), 'Predicciones_Simple'] = last_real_value

# Ajustar un modelo SARIMA a los datos históricos
sarima_model = pm.auto_arima(df['Peso_Promedio'].dropna(), seasonal=True, m=12, suppress_warnings=True)

# Realizar predicciones con el modelo SARIMA
sarima_predictions = sarima_model.predict(n_periods=len(future_df) + 1)  # Añade 1 para incluir la primera predicción

# Reemplazar la primera predicción del SARIMA con el último valor real
sarima_predictions[0] = last_real_value

# Añadir las predicciones del SARIMA al DataFrame
extended_df.loc[len(df)-1:, 'Predicciones_SARIMA'] = sarima_predictions

# Calcular el promedio de Peso_Promedio, FCA e IPS
average_weight = df['Peso_Promedio'].mean()
average_FCA = df['FCA'].mean()
average_ips = df['IPS'].mean()

# Crear el KPI para Peso_Promedio
kpi_weight = dbc.Card([
    dbc.CardBody([
        html.H4("Promedio de Peso_Promedio", className="card-title"),
        html.P(f"{average_weight:.2f}", className="card-text"),
    ])
], style={
    "width": "18rem", 
    "height": "6rem",
    "margin": "0 auto", 
    "border": "3px solid black",
    "background-color": "#f0f0f0"
})

# Crear el KPI para FCA
kpi_FCA = dbc.Card([
    dbc.CardBody([
        html.H4("Promedio de FCA", className="card-title"),
        html.P(f"{average_FCA:.2f}", className="card-text"),
    ])
], style={
    "width": "18rem", 
    "height": "6rem",
    "margin": "0 auto", 
    "border": "3px solid black",
    "background-color": "#f0f0f0"
})

# Crear el KPI para IPS
kpi_ips = dbc.Card([
    dbc.CardBody([
        html.H4("Promedio de IPS", className="card-title"),
        html.P(f"{average_ips:.2f}", className="card-text"),
    ])
], style={
    "width": "18rem", 
    "height": "6rem",
    "margin": "0 auto", 
    "border": "3px solid black",
    "background-color": "#f0f0f0"
})

# Eliminar filas duplicadas en la columna "Peso_Promedio"
df_unique = df.drop_duplicates(subset=['Peso_Promedio'])

# Seleccionar cada séptima fila
df_weekly = df.drop_duplicates(subset=['Peso_Promedio']).iloc[::7]

# Crear la tabla de datos
data_table = dash_table.DataTable(
    id='table',
    columns=[{
        "name": i, 
        "id": i, 
        "type": "numeric", 
        "format": Format(precision=2, scheme=Scheme.fixed)} for i in df_weekly.columns if i != "Unnamed: 0"],  
    data=df_weekly.round(2).to_dict('records'),
    style_table={'height': '300px', 'overflowY': 'auto'}
)

app.layout = html.Div([
    html.Div([
        dcc.DatePickerRange(
            id='my-date-picker-range',
            min_date_allowed=df['Fecha'].min(),
            max_date_allowed=future_df['Fecha'].max(),
            start_date=df['Fecha'].min(),
            end_date=future_df['Fecha'].max()
        ),
        html.Div(id='output-container-date-picker-range'),
        dcc.Dropdown(
            id='model-type',
            options=[{'label': 'Regresión lineal múltiple', 'value': 'multiple'},
                     {'label': 'Regresión lineal simple', 'value': 'simple'},
                     {'label': 'SARIMA', 'value': 'sarima'}],
            value='multiple',
        ),
    ], style={
        'width': '30%', 
        'display': 'inline-block', 
        'vertical-align': 'top',
        'border': '3px solid red',  # Añade un borde de color
        'padding': '10px',  # Añade un poco de espacio alrededor de los controles
        'margin-bottom': '10px',  # Añade un poco de espacio debajo del cuadro
    }),

    html.Div([
        html.Div([  # Contenedor para los KPIs
            kpi_weight,
            kpi_FCA,
            kpi_ips
        ], style={'display': 'flex', 'justify-content': 'space-between'}),  # Alinea los KPIs horizontalmente
        data_table,  # Añade la tabla de datos aquí
        dcc.Graph(id="graph"),
    ], style={'width': '68%', 'float': 'right', 'display': 'inline-block'}),
])

@app.callback(
    [Output('output-container-date-picker-range', 'children'),
     Output('graph', 'figure')],
    [Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'),
     Input('model-type', 'value')])
def update_output(start_date, end_date, model_type):
    string_prefix = 'You have selected: '
    if start_date and end_date:
        date_range_df = extended_df[(extended_df['Fecha'] >= start_date) & (extended_df['Fecha'] <= end_date)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Fecha'],
            y=df['Peso_Promedio'],
            mode='lines',
            name='Datos reales'
        ))
        if model_type == 'multiple':
            fig.add_trace(go.Scatter(
                x=date_range_df['Fecha'][len(df):],
                y=date_range_df['Peso_Promedio'][len(df):],
                mode='lines',
                name='Predicciones Múltiples'
            ))
        elif model_type == 'simple':
            fig.add_trace(go.Scatter(
                x=date_range_df['Fecha'][len(df):],
                y=date_range_df['Predicciones_Simple'][len(df):],
                mode='lines',
                name='Predicciones Simples'
            ))
        elif model_type == 'sarima':
            fig.add_trace(go.Scatter(
                x=date_range_df['Fecha'][len(df)-1:],
                y=date_range_df['Predicciones_SARIMA'][len(df)-1:],
                mode='lines',
                name='Predicciones SARIMA'
            ))
        fig.update_layout(title='Datos Reales y Predicciones de Peso Promedio',
                          xaxis_title='Fecha',
                          yaxis_title='Peso Promedio',
                          margin=dict(l=100, r=100, b=100, t=100, pad=4))
        return string_prefix + start_date + ' to ' + end_date, fig

    return 'No dates selected', go.Figure()


if __name__ == '__main__':
    app.run_server(debug=True)




