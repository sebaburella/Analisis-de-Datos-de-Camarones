import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from dash_table import Format
from dash_table.Format import Format, Scheme
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
import pmdarima as pm
import numpy as np
import plotly.express as px

app = dash.Dash(__name__)

# Importar el DataFrame
df = pd.read_csv('./datos/Pis_13_26.csv')
# Creamos otro dataframe igual para almacenar los datos de costo
df_costo = df.copy()

# Precicciones de Peso_Promedio
df['Fecha'] = pd.to_datetime(df['Fecha'])
# Crear una columna llamada Numero de dias
df['Numero de dias'] = (df['Fecha'] - df['Fecha'].min()).dt.days + 1
# Lista de columnas que se utilizarán en el modelo
feature_cols = ['Incremento', 'FCA', 'Sobrevivencia', 'Biomasa', 'Biomasa_Hectarea']
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
last_real_date = df['Fecha'].iloc[-1]
# Combinar el DataFrame original con el DataFrame de predicciones
extended_df = pd.concat([df, future_df], ignore_index=True)
# Adicionalmente, asegúrate de que la primera fecha de predicción sea la misma que la última fecha real
extended_df.loc[len(df), 'Fecha'] = last_real_date
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
# Añadir el último valor real a las predicciones de regresión lineal múltiple
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
extended_df.loc[len(df), 'Predicciones_Simple'] = last_real_value
# Ajustar un modelo SARIMA a los datos históricos
sarima_model = pm.auto_arima(df['Peso_Promedio'].dropna(), seasonal=True, m=12, suppress_warnings=True)
# Realizar predicciones con el modelo SARIMA
sarima_predictions = sarima_model.predict(n_periods=len(future_df))
# Añadir una fila con el último valor real al DataFrame de predicciones
last_real_row = pd.DataFrame({'Fecha': [last_real_date], 'Peso_Promedio': [last_real_value]})
future_df = pd.concat([last_real_row, future_df], ignore_index=True)
# Reemplazar la primera predicción del SARIMA con el último valor real
sarima_predictions = np.insert(sarima_predictions, 0, last_real_value)
# Eliminar el último valor que representa el valor real adicional
sarima_predictions = sarima_predictions[:-1]
# Añadir las predicciones del SARIMA al DataFrame
extended_df.loc[len(df):, 'Predicciones_SARIMA'] = sarima_predictions

# Predicciones de Costo_Directo
# Preprocesamiento de datos
df_costo['Sobrevivencia'] = df_costo['Sobrevivencia'].astype(float) * 100
df_costo['Fecha'] = pd.to_datetime(df_costo['Fecha'])
df_costo = df_costo.sort_values(by='Fecha')
# Crear una característica numérica para la fecha
df_costo['Dias_Desde_Inicio'] = (df_costo['Fecha'] - df_costo['Fecha'].min()).dt.days
# Realizar la regresión lineal
X = df_costo[['Dias_Desde_Inicio']]
y = df_costo['Costo_Directo']
model = LinearRegression()
model.fit(X, y)
# Generar fechas futuras para la predicción
dias_prediccion = 30
fecha_maxima = df_costo['Fecha'].max()
fechas_futuras = [fecha_maxima + timedelta(days=i) for i in range(1, dias_prediccion + 1)]
dias_desde_inicio_futuro = [(fecha - df_costo['Fecha'].min()).days for fecha in fechas_futuras]
# Realizar la predicción para las fechas futuras
predicciones = model.predict(pd.DataFrame(dias_desde_inicio_futuro, columns=['Dias_Desde_Inicio']))
# Tomar el último valor real y usarlo como primera predicción
ultimo_valor_real = df_costo.iloc[-1]['Costo_Directo']
predicciones = [ultimo_valor_real] + list(predicciones)
# Crear un DataFrame con las predicciones y fechas futuras
predicciones_df_costo = pd.DataFrame({'Fecha': [df_costo['Fecha'].max()] + fechas_futuras, 'Costo_Directo_Predicho': predicciones})
# Combinar las predicciones con el DataFrame original para graficar
df_costo_predicciones = pd.concat([df_costo, predicciones_df_costo])


# Calcular el promedio de Peso_Promedio, FCA e
average_weight = df['Peso_Promedio'].mean()
average_FCA = df['FCA'].mean()


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


# Eliminar filas duplicadas en la columna "Peso_Promedio"
df_unique = df.drop_duplicates(subset=['Peso_Promedio'])

# Seleccionar cada séptima fila
df_weekly = df.drop_duplicates(subset=['Peso_Promedio']).iloc[::7]

# Convertir la columna "Fecha" al formato deseado "YYYY-MM-DD"
df_weekly['Fecha'] = df_weekly['Fecha'].dt.strftime('%Y-%m-%d')


# Crear la tabla de datos
data_table = dash_table.DataTable(
    id='table',
    columns=[
        {"name": "Fecha", "id": "Fecha", "type": "datetime", "format": Format(precision=0, scheme=Scheme.fixed)}, # Cambiar el tipo a "datetime"
        {"name": "Incremento", "id": "Incremento", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)},
        {"name": "FCA", "id": "FCA", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)},
        {"name": "Sobrevivencia", "id": "Sobrevivencia", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)},
        {"name": "Biomasa", "id": "Biomasa", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)},
        {"name": "Biomasa_Hectarea", "id": "Biomasa_Hectarea", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)}
    ],
    data=df_weekly.round(2).to_dict('records'),
    style_table={'height': '100px', 'overflowY': 'auto'}
)

app.layout = html.Div([
    html.Div([
        dcc.DatePickerSingle(
            id='my-date-picker-single',
            min_date_allowed=df['Fecha'].min().strftime('%Y-%m-%d'),
            max_date_allowed=future_df['Fecha'].max().strftime('%Y-%m-%d'),
            initial_visible_month=df['Fecha'].min(),
            date=future_df['Fecha'].max()
        ),
        dcc.RadioItems(
            id='variable-objetivo',
            options=[
                {'label': 'Peso_Promedio', 'value': 'Peso_Promedio'},
                {'label': 'Costo_Directo', 'value': 'Costo_Directo'}
            ],
            value='Peso_Promedio',  # Valor predeterminado
            labelStyle={'display': 'block'}
        ),
        html.Div(id='output-container-date-picker-single'),
        dcc.Dropdown(
            id='model-type',
            options=[
                {'label': 'Regresión lineal múltiple', 'value': 'multiple'},
                {'label': 'Regresión lineal simple', 'value': 'simple'},
                {'label': 'SARIMA', 'value': 'sarima'}
            ],
            value='multiple',
        ),
    ], style={
        'width': '30%',
        'display': 'inline-block',
        'vertical-align': 'top',
        'border': '3px solid red',
        'padding': '5px',
        'margin-bottom': '10px',
    }),
    dcc.Graph(id='graph', style={'height': '400px'}),
    dcc.Graph(id='prediccion-costos-combinado', style={'height': '400px'}),
    html.Div(id='prediction-table'),
    dcc.Graph(id='pie-chart'),  # Agregar el componente del gráfico de torta
    dcc.Graph(id='bar-chart'),  # Agregar el componente del gráfico de columnas
])


@app.callback(
    [
        Output('output-container-date-picker-single', 'children'),
        Output('graph', 'figure'),
        Output('prediction-table', 'children'),
        Output('pie-chart', 'figure'),
        Output('bar-chart', 'figure'),
        Output('prediccion-costos-combinado', 'figure'),
    ],
    [
        Input('my-date-picker-single', 'date'),
        Input('model-type', 'value'),
        Input('variable-objetivo', 'value'),
    ])
def update_graphs(selected_date, model_type, variable_objetivo):
    string_prefix = 'You have selected: '
    if selected_date:
        date_range_df = extended_df[extended_df['Fecha'] <= selected_date]
        fig = go.Figure()

        if variable_objetivo == 'Peso_Promedio':
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
                    name='Predicciones Regresión Lineal Simple'
                ))
            elif model_type == 'sarima':
                fig.add_trace(go.Scatter(
                    x=date_range_df['Fecha'][len(df):],
                    y=date_range_df['Predicciones_SARIMA'][len(df):],
                    mode='lines',
                    name='Predicciones SARIMA'
                ))
            fig.update_layout(title='Datos Reales y Predicciones de Peso Promedio',
                              xaxis_title='Fecha',
                              yaxis_title='Peso Promedio',
                              margin=dict(l=100, r=100, b=100, t=100, pad=4))
        elif variable_objetivo == 'Costo_Directo':
            fig = grafico_costo()  # Mostrar el gráfico de costos
            prediction_table = update_prediction_table(df_costo_predicciones)  # Usar el DataFrame de costos para la tabla de predicciones
            pie_chart = px.pie(values=df['Comentario'].value_counts().values,
                           names=df['Comentario'].value_counts().index,
                           title='Distribución de Comentarios')  
            bar_chart = px.bar(df, x='Fecha', y='Costo_Directo', title='Gráfico de Columnas')  # Puedes mantenerlo vacío para el caso de Costo_Directo
            costo_fig = grafico_costo()  # Actualizar el gráfico de predicciones de Costo Directo

            return string_prefix + selected_date, fig, prediction_table, pie_chart, bar_chart, costo_fig

        # Llamamos a la función de actualización de la tabla de predicciones
        prediction_table = update_prediction_table(date_range_df)

        # Agregar el gráfico de torta
        pie_chart = px.pie(values=df['Comentario'].value_counts().values,
                           names=df['Comentario'].value_counts().index,
                           title='Distribución de Comentarios')

        # Agregar el gráfico de columnas
        bar_chart = px.bar(df, x='Fecha', y='Peso_Promedio', title='Gráfico de Columnas')

        # Agregar el gráfico de predicciones de Costo Directo
        costo_fig = grafico_costo()

        return string_prefix + selected_date, fig, prediction_table, pie_chart, bar_chart, costo_fig

    return 'No date selected', go.Figure(), '', px.pie(), px.bar(), go.Figure()


@app.callback(
    Output('model-type', 'options'),
    Input('variable-objetivo', 'value')
)
def update_model_options(variable_objetivo):
    if variable_objetivo == 'Costo_Directo':
        return [{'label': 'Regresión lineal simple', 'value': 'simple'}]
    else:
        return [
            {'label': 'Regresión lineal múltiple', 'value': 'multiple'},
            {'label': 'Regresión lineal simple', 'value': 'simple'},
            {'label': 'SARIMA', 'value': 'sarima'},
        ]



def grafico_costo():
    # Crear la figura para el gráfico con plotly
    fig = go.Figure()
    # Agregar la línea de los datos reales
    fig.add_trace(go.Scatter(x=df_costo_predicciones['Fecha'], y=df_costo_predicciones['Costo_Directo'],
                             mode='lines', name='Datos reales'))
    # Agregar la línea de las predicciones
    fig.add_trace(go.Scatter(x=df_costo_predicciones['Fecha'], y=df_costo_predicciones['Costo_Directo_Predicho'],
                             mode='lines', name='Predicciones', line=dict(dash='dash')))
    # Configurar el diseño del gráfico
    fig.update_layout(
        xaxis=dict(type='date'),
        yaxis=dict(title='Costo Directo'),
        title='Predicción de Costo Directo para Fechas Futuras'
    )
    return fig



# Definimos la función que actualiza la tabla de predicciones
def update_prediction_table(dataframe):
    table = dash_table.DataTable(
        data=dataframe.round(2).to_dict('records'),
        columns=[
            {'name': col, 'id': col} for col in dataframe.columns
        ],
        style_table={'height': '200px', 'overflowY': 'auto'},
    )
    return table


if __name__ == '__main__':
    app.run_server(debug=True)







