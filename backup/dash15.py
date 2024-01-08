import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from dash import dash_table
from dash.dash_table.Format import Format
from dash.dash_table.Format import Format, Scheme
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px

# Crear la aplicación de Dash
app = dash.Dash(__name__)

# Importar el DataFrame
df = pd.read_csv('./datos/Pis_11_45.csv')

# Creamos otro dataframe igual para almacenar los datos de costo
df_costo = df.copy()

# Crear la columna Costo_Directo_Acumulado
df_costo['Costo_Directo_Acumulado'] = df_costo['Costo_Directo'].cumsum()
df_costo['Costo_Difference'] = df_costo['Costo_Directo'].diff()

# Calcular la correlación entre las columnas
correlation = df_costo['Costo_Directo'].corr(df_costo['Sobrevivencia'])

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
#calcular el r2 score
r2_score = final_model.score(features[:len(df)], df['Peso_Promedio'].dropna())
# Realizar las predicciones
predictions = final_model.predict(features) +1.5

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
#calcular el r2 score
r2_score_simple = simple_model.score(df[[max_corr_feature]], df['Peso_Promedio'].dropna())

# Realizar las predicciones
extended_df['Predicciones_Simple'] = simple_model.predict(extended_df[[max_corr_feature]])
extended_df.loc[len(df), 'Predicciones_Simple'] = last_real_value

# Añadir una fila con el último valor real al DataFrame de predicciones
last_real_row = pd.DataFrame({'Fecha': [last_real_date], 'Peso_Promedio': [last_real_value]})
future_df = pd.concat([last_real_row, future_df], ignore_index=True)

# Predicciones de Costo_Directo_Acumulado
df_costo.dropna(subset=['Costo_Directo_Acumulado'], inplace=True)

df_costo['Sobrevivencia'] = df_costo['Sobrevivencia'].astype(float) * 100
df_costo['Fecha'] = pd.to_datetime(df_costo['Fecha'])
df_costo = df_costo.sort_values(by='Fecha')
df_costo['Dias_Desde_Inicio'] = (df_costo['Fecha'] - df_costo['Fecha'].min()).dt.days
X = df_costo[['Dias_Desde_Inicio']]
y = df_costo['Costo_Directo_Acumulado']
X = X[y.notna()]
y = y.dropna()
# Entrenando el modelo de regresión lineal simple para predecir el costo directo acumulado
model = LinearRegression()
model.fit(X, y)
#calcular el r2 score
r2_score_costo_simple = model.score(X, y)
# Prediciendo para los próximos 30 días
dias_prediccion = 70
fecha_maxima = df_costo['Fecha'].max()
fechas_futuras = [fecha_maxima + timedelta(days=i) for i in range(1, dias_prediccion + 1)]
dias_desde_inicio_futuro = [(fecha - df_costo['Fecha'].min()).days for fecha in fechas_futuras]
predicciones = model.predict(pd.DataFrame(dias_desde_inicio_futuro, columns=['Dias_Desde_Inicio']))
# Agregando el último valor real como la primera predicción
ultimo_valor_real = df_costo.iloc[-1]['Costo_Directo_Acumulado']
predicciones_cumulativas = np.concatenate(([ultimo_valor_real], predicciones))
# Creando un DataFrame para las predicciones
predicciones_df_costo = pd.DataFrame({'Fecha': [df_costo['Fecha'].max()] + fechas_futuras, 'Costo_Directo_Acumulado_Predicho': predicciones_cumulativas})
df_costo_predicciones = pd.concat([df_costo, predicciones_df_costo])

# Filtrar los datos donde Peso_Promedio sea mayor a 20
filtered_data = df[df["Peso_Promedio"] > 20].copy()
# Calcular la diferencia del Costo_Directo con respecto al día anterior
filtered_data["Costo_Difference"] = filtered_data["Costo_Directo"].diff()
# Filtrar aquellos registros donde el costo está disminuyendo
decreasing_cost_data = filtered_data[filtered_data["Costo_Difference"] < 0]
# Punto óptimo de cosecha
optimal_date = decreasing_cost_data['Fecha'].iloc[0]
optimal_cost = df_costo.loc[decreasing_cost_data.index[0], 'Costo_Directo_Acumulado'] # Usa df_costo para obtener el costo acumulado
optimal_weight = decreasing_cost_data['Peso_Promedio'].iloc[0]

# Eliminar filas duplicadas en la columna "Peso_Promedio"
df_unique = df.drop_duplicates(subset=['Peso_Promedio'])
# Seleccionar cada séptima fila
df_weekly = df.drop_duplicates(subset=['Peso_Promedio']).iloc[::7]
# Convertir la columna "Fecha" al formato deseado "YYYY-MM-DD"
df_weekly['Fecha'] = df_weekly['Fecha'].dt.strftime('%Y-%m-%d')


def create_optimal_harvest_graph():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Fecha'],
        y=df['Peso_Promedio'],
        name='Peso Promedio',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=df_costo['Fecha'],
        y=df_costo['Costo_Directo_Acumulado'],
        name='Costo Directo Acumulado',
        line=dict(color='green'),
        yaxis='y2'
    ))

    fig.update_layout(
        title='Variación del Peso Promedio y Costo Directo Acumulado a lo largo del tiempo',
        xaxis=dict(title='Fecha'),
        yaxis=dict(title='Peso Promedio'),
        yaxis2=dict(title='Costo Directo Acumulado', overlaying='y', side='right')
    )

    return fig

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
#app.layout pero agregando el r2 score

app.layout = html.Div([
    html.Link(
        rel='stylesheet',
        href='/static/acosux.css'
    ),
    html.H1('Dashboard de Acuicultura'),
    html.Div(id='r2-score-display', children=f"R2 Score: {r2_score:.4f}"),
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
                {'label': 'Regresión lineal simple', 'value': 'simple'}
            ],
            value='multiple',
        ),
    ], className='sidebar'),
    dcc.Graph(id='graph', style={'height': '400px'}),
    dcc.Graph(figure=create_optimal_harvest_graph(), id='optimal-harvest-graph'),  # Gráfico del punto óptimo de cosecha
    html.Div(id='prediction-table'),
    dcc.Graph(id='pie-chart', className='chart'),  # Agregar el componente del gráfico de torta
    dcc.Graph(id='bar-chart', className='chart',style={'height': '350px'}),  # Agregar el componente del gráfico de columnas
])



@app.callback(
    [
        Output('output-container-date-picker-single', 'children'),
        Output('graph', 'figure'),
        Output('pie-chart', 'figure'),
        Output('bar-chart', 'figure'),
        Output('prediction-table', 'children'),
        Output('r2-score-display', 'children')  # Asegúrate de haber añadido esta línea
    ],
    [
        Input('my-date-picker-single', 'date'),
        Input('model-type', 'value'),
        Input('variable-objetivo', 'value'),
    ])



def update_components(selected_date, model_type, variable_objetivo):
    # Ensure date_range_df is defined and populated
    date_range_df = extended_df[extended_df['Fecha'] <= selected_date]

    string_prefix = 'You have selected: '
    if not selected_date:
        return string_prefix + 'No se ha seleccionado ninguna fecha', go.Figure(), px.pie(), px.bar()

    formatted_dates = [date.strftime("%d-%m-%Y") for date in date_range_df['Fecha']]

    prediction_table = dash_table.DataTable(
        id='prediction-table',
        columns=[
            {"name": "Fecha", "id": "Fecha", "type": "text"},
            {"name": "Predicciones Múltiples", "id": "Peso_Promedio", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)},
            {"name": "Predicciones Regresión Lineal Simple", "id": "Predicciones_Simple", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)}
        ],
        data=[{
            'Fecha': formatted_date,
            'Peso_Promedio': peso_promedio,
            'Predicciones_Simple': pred_simple
        } for formatted_date, peso_promedio, pred_simple in zip(formatted_dates, date_range_df['Peso_Promedio'][len(df):], date_range_df['Predicciones_Simple'])],
        style_table={'height': '100px', 'overflowY': 'auto'}
    )

    fig = go.Figure()
    pie_chart = px.pie()
    bar_chart = px.bar()

    if variable_objetivo == 'Peso_Promedio':
        fig.add_trace(go.Scatter(
            x=df['Fecha'],
            y=df['Peso_Promedio'],
            mode='lines',
            name='Datos reales'
        ))
        if model_type == 'multiple':
            r2_score_value = final_model.score(features[:len(df)], df['Peso_Promedio'].dropna())
            r2_display = f"R2 Score (Multiple): {r2_score_value:.4f}"
            fig.add_trace(go.Scatter(
                x=date_range_df['Fecha'][len(df):],
                y=date_range_df['Peso_Promedio'][len(df):],
                mode='lines',
                name='Predicciones Múltiples'
            ))
        elif model_type == 'simple':
            r2_score_value_simple = simple_model.score(df[[max_corr_feature]], df['Peso_Promedio'].dropna())
            r2_display = f"R2 Score (Simple): {r2_score_value_simple:.4f}"
            fig.add_trace(go.Scatter(
                x=date_range_df['Fecha'][len(df):],
                y=date_range_df['Predicciones_Simple'][len(df):],
                mode='lines',
                name='Predicciones Regresión Lineal Simple'
            ))
        fig.update_layout(title='Datos Reales y Predicciones de Peso Promedio',
                        xaxis_title='Fecha',
                        yaxis_title='Peso Promedio',
                        margin=dict(l=100, r=100, b=100, t=100, pad=4))
        # Agregar el componente del gráfico de torta
        pie_chart = px.pie(values=df_unique['Comentario'].value_counts().values,
                        names=df_unique['Comentario'].value_counts().index,
                        title='Distribución de Comentarios')
        # Agregar el componente del gráfico de columnas
        bar_chart = px.bar(df_weekly, x='Fecha', y='Peso_Promedio', title='Promedio de Peso_Promedio Semanal')

    elif variable_objetivo == 'Costo_Directo':
        fig = grafico_costo(selected_date)  # Actualizar el gráfico de predicciones de Costo Directo
        r2_score_value_costo = model.score(X, y)
        r2_display = f"R2 Score (Costo Simple): {r2_score_value_costo:.4f}"
        pie_chart = px.pie(values=df['Comentario'].value_counts().values,
                           names=df['Comentario'].value_counts().index,
                           title='Distribución de Comentarios')
        bar_chart = px.bar(df, x='Fecha', y='Costo_Directo', title='Gráfico de Columnas')

    return string_prefix + selected_date, fig, pie_chart, bar_chart, prediction_table, 'Precision :'+r2_display



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
            {'label': 'Regresión lineal simple', 'value': 'simple'}
        ]


def grafico_costo(selected_date):
    date_range_df_costo = df_costo_predicciones[df_costo_predicciones['Fecha'] <= selected_date] 
    fig = go.Figure()   
    # Aquí es donde hacemos el cambio clave: graficamos Costo_Directo_Acumulado en lugar de Costo_Directo
    fig.add_trace(go.Scatter(x=date_range_df_costo['Fecha'], y=date_range_df_costo['Costo_Directo_Acumulado'],
                             mode='lines', name='Datos reales'))   
    fig.add_trace(go.Scatter(x=date_range_df_costo['Fecha'], y=date_range_df_costo['Costo_Directo_Acumulado_Predicho'],
                             mode='lines', name='Predicciones', line=dict(dash='dash')))
    fig.update_layout(
        xaxis=dict(type='date'),
        yaxis=dict(title='Costo Directo Acumulado'),
        title='Predicción de Costo Directo Acumulado para Fechas Futuras'
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)






