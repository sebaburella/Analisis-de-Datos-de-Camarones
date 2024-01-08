import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from dash import dash_table
import dash_bootstrap_components as dbc
from dash.dash_table.Format import Format, Scheme
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
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

#Creamos las predicciones de peso promedio
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


###############     PUNTO OPTIMO DE COSECHA     #####################
price_per_size = {
    'talla 20/30': 2.49,
    'talla 30/40': 1.91,
    'talla 40/50': 1.72,
    'talla 50/60': 1.63,
    'talla 60/70': 1.54,
    'talla 70/80': 1.22,
    'talla 80/100': 1.22
}
# Función para categorizar las tallas
def categorize_talla_units_per_kilo(peso_promedio):
    units_per_kilo = 1000 / peso_promedio if peso_promedio > 0 else 0
    if units_per_kilo >= 20 and units_per_kilo < 30:
        return 'talla 20/30'
    elif units_per_kilo >= 30 and units_per_kilo < 40:
        return 'talla 30/40'
    elif units_per_kilo >= 40 and units_per_kilo < 50:
        return 'talla 40/50'
    elif units_per_kilo >= 50 and units_per_kilo < 60:
        return 'talla 50/60'
    elif units_per_kilo >= 60 and units_per_kilo < 70:
        return 'talla 60/70'
    elif units_per_kilo >= 70 and units_per_kilo < 80:
        return 'talla 70/80'
    elif units_per_kilo >= 80:
        return 'talla 80/100'
    else:
        return 'talla desconocida'
# Eliminar outliers
Q1 = df['Peso_Promedio'].quantile(0.25)
Q3 = df['Peso_Promedio'].quantile(0.75)
IQR = Q3 - Q1
filtered_df = df[(df['Peso_Promedio'] >= Q1 - 1.5 * IQR) & (df['Peso_Promedio'] <= Q3 + 1.5 * IQR)]
# Categorizar las tallas en el dataset filtrado
filtered_df['Talla'] = filtered_df['Peso_Promedio'].apply(categorize_talla_units_per_kilo)
# Calcular el número de camarones por hectárea teniendo en cuenta la reducción
camarones_por_hectarea = 8.87 * 10000 * 7
reduction_factor = 0.25  # Nuevo factor de reducción
reduced_camarones_por_hectarea = camarones_por_hectarea * reduction_factor
# Calcular el precio de venta aproximado en el dataset filtrado
filtered_df['Precio_Venta_Aprox'] = filtered_df['Talla'].apply(lambda x: price_per_size.get(x, 0)) * reduced_camarones_por_hectarea
# Añadir una fluctuación aleatoria al Precio de Venta Aproximado Diario
np.random.seed(0)  # Para reproducibilidad
fluctuation = np.random.uniform(-0.02, 0.02, len(filtered_df))
filtered_df['Ganancias_Aprox'] = filtered_df['Precio_Venta_Aprox'] * (1 + fluctuation)
# Aplicar el nuevo factor de reducción
filtered_df['Ganancias_Aprox'] = filtered_df['Ganancias_Aprox'] * reduction_factor - (filtered_df['Costo_Directo'] + filtered_df['Costo_Indirecto'])
# A las Ganancias estimadas anteriores les restamos la desviacion estandar de las ganancias
filtered_df['Ganancias_Aprox'] = filtered_df['Ganancias_Aprox'] - filtered_df['Ganancias_Aprox'].std() * 3
# Encontrar el punto máximo del Precio de Venta Aproximado
max_index = filtered_df['Ganancias_Aprox'].idxmax()
max_date = filtered_df.loc[max_index, 'Fecha']
max_price = filtered_df.loc[max_index, 'Ganancias_Aprox']





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
    html.Link(
        rel='stylesheet',
        href='/assets/style.css'  # Asegúrate de que la ruta coincida con la ubicación de tu archivo CSS
    ),
    html.Div([
        html.Header([
            html.H1('Ganancias Aproximadas', className='main-title'),
        ]),
        html.Main([
            dcc.Graph(
                id='ganancias-aproximadas',
                config={'displayModeBar': False},
            ),
            html.H2('Dashboard de Acuicultura', className='dashboard-title'),
            html.Div(id='r2-score-display', className='r2-score'),
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
            html.Div(id='prediction-table', className='prediction-table'),
            dcc.Graph(id='pie-chart', className='chart'),
            dcc.Graph(id='bar-chart', className='chart', style={'height': '350px'}),
        ], className='content'),
    ], className='container'),
], style={'background-color': 'white'})  # Cambiar el fondo principal de la página a blanco








#############      PUNTO OPTIMO DE COSECHA        #####################
@app.callback(
    Output('ganancias-aproximadas', 'figure'),
    Input('ganancias-aproximadas', 'relayoutData'),
)
def grafico_punto_optimo_cosecha(relayoutData):
    # Gráfico del Precio de Venta Aproximado diario con fluctuación y reducción
    trace = go.Scatter(
        x=filtered_df['Fecha'],
        y=filtered_df['Ganancias_Aprox'],
        mode='lines',
        name='Ganancias Aproximadas',
    )
    
    # Nueva área con la misma tendencia pero con opacidad del 20%
    new_trace = go.Scatter(
        x=filtered_df['Fecha'],
        y=filtered_df['Ganancias_Aprox'],
        fill='tozeroy',  # Esto rellena el área desde el valor y hasta el eje x
        mode='none',  # Solo queremos el área, no la línea ni los marcadores
        fillcolor='rgba(255, 165, 0, 0.2)',  # Color de relleno con opacidad del 20%
        showlegend=False
    )

    # Punto óptimo de cosecha
    trace_max = go.Scatter(
        x=[max_date],
        y=[max_price],
        mode='markers',
        name='Punto Óptimo de Cosecha',
        marker=dict(
            color='red',
            size=10,
        ),
        text=['Punto Óptimo de Cosecha'],
    )

    layout = go.Layout(
        title='Ganancias Aproximadas',
        xaxis=dict(title='Fecha'),
        yaxis=dict(title='Precio de Venta Aproximado ($)'),
        annotations=[
            dict(
                x=max_date,
                y=max_price,
                text=' ',
                showarrow=False,
                arrowhead=7,
                ax=-30,
                ay=10,
            ),
        ],
    )

    return {'data': [new_trace, trace, trace_max], 'layout': layout}




#########      GRAFICO PREDICCIONES      ##################
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
def grafico_predicciones(selected_date, model_type, variable_objetivo):
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
        bar_chart = px.bar(df_weekly, x='Fecha', y='Peso_Promedio', title='Promedio de Peso_Promedio Semanal')
        bar_chart.update_traces(marker=dict(opacity=0.3, line=dict(color='black', width=1.0)))

    elif variable_objetivo == 'Costo_Directo':
        fig = grafico_costo(selected_date)  # Actualizar el gráfico de predicciones de Costo Directo
        r2_score_value_costo = model.score(X, y)
        r2_display = f"R2 Score (Costo Simple): {r2_score_value_costo:.4f}"
        pie_chart = px.pie(values=df['Comentario'].value_counts().values,
                           names=df['Comentario'].value_counts().index,
                           title='Distribución de Comentarios')
        bar_chart = px.bar(df, x='Fecha', y='Costo_Directo', title='Gráfico de Columnas')
        bar_chart.update_traces(marker=dict(opacity=0.3, line=dict(color='black', width=1.0)))
        #opacidad del grafico de columnas
        bar_chart.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return string_prefix + selected_date, fig, pie_chart, bar_chart, prediction_table, 'Precision :'+r2_display
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
