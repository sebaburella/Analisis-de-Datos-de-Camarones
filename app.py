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
from dash.exceptions import PreventUpdate
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Inicializar la aplicación Dash y especificar el tema oscuro de Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

server = app.server  # Exponer el servidor Flask para que Gunicorn pueda interactuar con él

df = pd.read_csv('./postgres2.csv')
#comprobamos si la columna fecha se encuentra en el dataframe
if 'Fecha' not in df.columns:
    print("ERROR: La columna 'Fecha' no se encuentra después de importar los datos.")
# Interpolation to impute missing values
df.interpolate(inplace=True)
# Drop any remaining rows with NaN values
df.dropna(subset=['Peso_Promedio'], inplace=True)

#REGRESION LINEAL SIMPLE 
# Creamos otro dataframe igual para almacenar los datos de costo
df_costo = df.copy()
# Crear la columna Costo_Directo_Acumulado
df_costo['Costo_Directo_Acumulado'] = df_costo['Costo_Directo'].cumsum()
df_costo['Costo_Indirecto_Acumulado'] = df_costo['Costo_Indirecto'].cumsum()
df_costo['Costo_Difference'] = df_costo['Costo_Directo'].diff()
# Calcular la correlación entre las columnas
correlation = df_costo['Costo_Directo'].corr(df_costo['Sobrevivencia'])
df['Fecha'] = pd.to_datetime(df['Fecha'])
# Crear una columna llamada Numero de dias
df['Numero de dias'] = (df['Fecha'] - df['Fecha'].min()).dt.days + 1
# Lista de columnas que se utilizarán en el modelo
feature_cols = ['Biomasa', 'Biomasa_Hectarea']
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
date_range_df = pd.DataFrame()

date_range_df['Predicciones_Simple'] = future_df['Biomasa']
last_real_date = df['Fecha'].iloc[-1]
# Combinar el DataFrame original con el DataFrame de predicciones
extended_df = pd.concat([df, future_df], ignore_index=True)
# Adicionalmente, asegúrate de que la primera fecha de predicción sea la misma que la última fecha real
extended_df.loc[len(df), 'Fecha'] = last_real_date


# REGRESION LINEAL MULTIPLE
# Features and target variable definition
features = ['Costo_Directo', 'Biomasa', 'Balanceado', 'Sobrevivencia']
# Removing rows with NaN values in the selected features and target variable
df_cleaned = df.dropna(subset=['Peso_Promedio'] + features)
# Splitting the features and target variable
X = df_cleaned[features]
y = df_cleaned['Peso_Promedio']
# Training the multiple linear regression model
model_multiple = LinearRegression()
model_multiple.fit(X, y)
# Creating a DataFrame for future predictions
future_df_multiple = pd.DataFrame()
future_df_multiple['Fecha'] = future_dates
future_df_multiple['Numero de dias'] = (future_df_multiple['Fecha'] - df['Fecha'].min()).dt.days
# Imputing the features for future dates using the last observed value
for feature in features:
    future_df_multiple[feature] = df_cleaned[feature].iloc[-1]
# Using the multiple linear regression model to predict Peso_Promedio for the future dates
future_df_multiple['Peso_Promedio'] = model_multiple.predict(future_df_multiple[features])




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

# Convertir el diccionario price_per_size a DataFrame
price_df = pd.DataFrame(list(price_per_size.items()), columns=['Talla', 'Precio'])
# Crear el componente DataTable
price_table = dash_table.DataTable(
    id='price-table',
    columns=[{"name": i, "id": i} for i in price_df.columns],
    data=price_df.to_dict('records'),
    style_table={'height': '150px', 'overflowY': 'auto'},
    style_cell={
        'backgroundColor': '#f8f9fa',
        'color': 'black',
        'border': '1px solid grey',
    }
)
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
filtered_df = df[(df['Peso_Promedio'] >= Q1 - 2 * IQR) & (df['Peso_Promedio'] <= Q3 + 2 * IQR)]
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

# Calcula la media de la columna Biomasa
media_biomasa = filtered_df['Biomasa'].mean()
# Calcula la desviación estándar de la población para la columna Biomasa
N = len(filtered_df)
desviacion_estandar_biomasa = np.sqrt(((filtered_df['Biomasa'] - media_biomasa) ** 2).sum() / N)
# Resta la desviación estándar calculada a la columna Ganancias_Aprox
filtered_df['Ganancias_Aprox'] = filtered_df['Ganancias_Aprox'] - desviacion_estandar_biomasa * 2

# Encontrar el punto máximo del Precio de Venta Aproximado
max_index = filtered_df['Ganancias_Aprox'].idxmax()
max_date = filtered_df.loc[max_index, 'Fecha']
max_price = filtered_df.loc[max_index, 'Ganancias_Aprox']

# We also need the 'Peso_Promedio' value at max_index to determine the size category
peso_promedio_at_max = filtered_df.loc[max_index, 'Peso_Promedio']
talla_at_max = categorize_talla_units_per_kilo(peso_promedio_at_max)



# Eliminar filas duplicadas en la columna "Peso_Promedio"
df_unique = df.drop_duplicates(subset=['Peso_Promedio'])
# Seleccionar cada séptima fila
df_weekly = df.drop_duplicates(subset=['Peso_Promedio']).iloc[::7]
# Convertir la columna "Fecha" al formato deseado "YYYY-MM-DD"
df_weekly['Fecha'] = df_weekly['Fecha'].dt.strftime('%Y-%m-%d')

#creamos una columna del Costo Directo acumulado
df['Costo_Directo_Acumulado'] = df['Costo_Directo'].cumsum()


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
        html.Div([
            html.H1(
                'Analisis de Datos de Camarones',
                style={
                    'fontSize': 40,
                    'textAlign': 'center',
                    'textDirection': 'vertical',
                    'dir': 'rtl',
                    'padding': '20px',
                    'padding-top': '70px',
                    'color': 'white',
                    'margin-left': 'auto',
                    'margin-right': 'auto'
                },
                className='eight columns'
            ),
            html.Img(
                src="/static/ACOSUX.png",
                className='four columns',
                style={
                    'height': '20%',
                    'width': '20%',
                    'float': 'right',
                    'position': 'relative',
                    'margin-top': -280,
                    'align': 'center',
                    'background-color': '#187da0',
                }
            ),
        ], className="row"),

        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H6("PRECISION DEL MODELO:", className="card-title",style={'color': '#41444b'}),
                        html.P(id="r2-score-display", className="card-text",style={'color': '#41444b'})
                    ]),
                    style={"background-color": "#f8f9fa", "box-shadow": "4px 4px 2.5px #dddddd", "border-radius": "15px"}
                ),
                width=2
            ),
            # Repite lo mismo para los demás indicadores
        ], align="center", justify="center"),

        html.Div([
            dbc.Row([
                dbc.Col([
                    html.H3('Fecha Final:'),
                    dcc.DatePickerSingle(
                        id='my-date-picker-single',
                        min_date_allowed=df['Fecha'].min().strftime('%Y-%m-%d'),
                        max_date_allowed=future_df['Fecha'].max().strftime('%Y-%m-%d'),
                        initial_visible_month=df['Fecha'].min(),
                        date=future_df['Fecha'].max()
                    ),
                    html.Div(id='output-container-date-picker-single'),
                    html.H3('Tipo de Modelo:'),
                    dbc.DropdownMenu(
                        id='model-type',
                        label="Selecciona un modelo",  
                        children=[
                            dbc.DropdownMenuItem('Regresión lineal múltiple', id='item-multiple'),
                            dbc.DropdownMenuItem('Regresión lineal simple', id='item-simple'),
                            dbc.DropdownMenuItem('Regresión lineal polinómica', id='item-polinomica')
                        ],
                    ),
                    
                    
                ], className='six columns', style={
                    'border-radius': '15px',
                    'backgroundColor': '#f8f9fa',
                    'box-shadow': '4px 4px 2.5px #dddddd',
                    'padding': '20px',
                    'margin-left': 'auto',
                    'margin-right': 'auto',
                    'margin-top': '25px',
                    'color': '#41444b'
                }),
            ], className='row justify-content-center'),

            dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id='graph-peso-promedio',  # ID para el gráfico de Peso_Promedio
                        style={
                            'border-radius': '15px',
                            'backgroundColor': '#f8f9fa',
                            'box-shadow': '4px 4px 2.5px #dddddd',
                            'padding': '20px',
                            'margin-left': 'auto',
                            'margin-right': 'auto',
                            'margin-top': '25px',
                        }
                    ),
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(
                        id='graph-costo-directo',  # ID para el gráfico de Costo_Directo
                        style={
                            'border-radius': '15px',
                            'backgroundColor': '#f8f9fa',
                            'box-shadow': '4px 4px 2.5px #dddddd',
                            'padding': '20px',
                            'margin-left': 'auto',
                            'margin-right': 'auto',
                            'margin-top': '25px',
                        }
                    ),
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(id='costos-acumulados',
                              figure={
                                  'data': [
                                      go.Scatter(
                                          x=df_costo['Fecha'],
                                          y=df_costo['Costo_Directo_Acumulado'],
                                          mode='lines',
                                          name='Costo Directo Acumulado'
                                      ),
                                      go.Scatter(
                                          x=df_costo['Fecha'],
                                          y=df_costo['Costo_Indirecto_Acumulado'],
                                          mode='lines',
                                          name='Costo Indirecto Acumulado'
                                      )
                                  ],
                                  'layout': go.Layout(
                                      title='Costos Directos e Indirectos Acumulados',
                                      xaxis={'title': 'Fecha'},
                                      yaxis={'title': 'Costo Acumulado'},
                                      height=450,  # Reducir la altura del gráfico
                                      margin=dict(l=30, r=30, b=30, t=30)  # Reducir márgenes
                                  )
                              },
                              style={
                                  'border-radius': '15px',
                                  'backgroundColor': '#f8f9fa',
                                  'box-shadow': '4px 4px 2.5px #dddddd',
                                  'padding': '20px',
                                  'margin-left': 'auto',
                                  'margin-right': 'auto',
                                  'margin-top': '25px',
                                  'margin-bottom': '25px',
                                  'width': '100%'  # Ancho del 100% para ocupar el espacio completo
                              }
                              ),
                    width=6
                ),
            
            
            
            
#GRAFICO GANANCIAS APROXIMADAS Y INDICADOR DE TALLA CAMARON   
  # Espacio entre los gráficos
            html.Div([
                html.H4("Punto Óptimo de Cosecha"),
                html.P(f"Fecha: {max_date}"),
                html.P(f"Ganancia Aproximada: ${max_price:.2f}"),
                html.Div(f"Talla del Camarón: {talla_at_max}", style={
                    'fontSize': '24px',
                    'color': 'white',
                    'backgroundColor': 'green',
                    'padding': '10px',
                    'borderRadius': '5px',
                    'textAlign': 'center'
                })
            ], style={'border': '1px solid black', 'padding': '20px', 'margin': '20px', 'backgroundColor': '#03254E', 'color': 'white'}),

            # Crear una fila para el gráfico y la tabla, uno al lado del otro
            
                dbc.Col(
                    dcc.Graph(id='ganancias-aproximadas', style={
                        'border-radius': '15px',
                        'backgroundColor': '#f8f9fa',
                        'box-shadow': '4px 4px 2.5px #dddddd',
                        'padding': '20px',
                        'margin-left': 'auto',
                        'margin-right': 'auto',
                        'margin-top': '25px'
                    }),
                    width=6
                ),
                dbc.Col(
                    html.Div([
                        html.H4("PRECIOS DE TALLAS"),
                        html.Div(f"Talla del Camarón: {talla_at_max}", style={
                            'fontSize': '24px',
                            'color': 'white',
                            'backgroundColor': 'green',
                            'padding': '10px',
                            'borderRadius': '5px',
                            'textAlign': 'center'
                        }),
                        dash_table.DataTable(
                            id='price-table',
                            columns=[
                                {"name": "Talla", "id": "Talla"},
                                {"name": "Precio", "id": "Precio"}
                            ],
                            data=[{"Talla": key, "Precio": value} for key, value in price_per_size.items()],
                            style_table={'height': 'auto', 'overflowY': 'auto', 'color': '#41444b'},
                        )
                    ], style={'border': '1px solid black', 'padding': '20px', 'margin': '20px', 'backgroundColor': '#03254E', 'color': 'white'}),
                    width=6
                ),
            ]),


            

            dbc.Row([
        

                dbc.Col([
                    dcc.Graph(id='bar-chart', style={
                        'border-radius': '15px',
                        'backgroundColor': '#f8f9fa',
                        'box-shadow': '4px 4px 2.5px #dddddd',
                        'padding': '20px',
                        'height': '300px'
                    }),
                    html.Div([
                        dash_table.DataTable(
                            id='data-table',
                            columns=[
                                {"name": "Fecha", "id": "Fecha", "type": "datetime", "format": Format(precision=0, scheme=Scheme.fixed)},
                                {"name": "Incremento", "id": "Incremento", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)},
                                {"name": "FCA", "id": "FCA", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)},
                                {"name": "Sobrevivencia", "id": "Sobrevivencia", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)},
                                {"name": "Biomasa", "id": "Biomasa", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)},
                                {"name": "Biomasa_Hectarea", "id": "Biomasa_Hectarea", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)}
                            ],
                            data=df_weekly.round(2).to_dict('records'),
                            style_table={'height': '300px', 'overflowY': 'auto','color': '#41444b'}
                        )
                    ], className='twelve columns', style={
                        'border-radius': '15px',
                        'backgroundColor': '#f8f9fa',
                        'box-shadow': '4px 4px 2.5px #dddddd',
                        'padding': '20px',
                        'margin-left': 'auto',
                        'margin-right': 'auto',
                        'margin-top': '25px',
                    }),
                ], width=6, id='bar-chart-container'),
            ], className='row justify-content-center'),

        ], className='ten columns offset-by-one')
    ])
])




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
        # Configurar el fondo del gráfico como transparente
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return {'data': [new_trace, trace, trace_max], 'layout': layout}




#########      GRAFICO PREDICCIONES      ##################

# Calculate correlation with 'Peso_Promedio' for each feature
features = df.drop(columns=['Peso_Promedio', 'Fecha'])

# Removing non-numeric columns for correlation calculation
features = features.select_dtypes(include=['float64', 'int64'])

correlations = features.corrwith(df['Peso_Promedio'])
max_corr_feature = correlations.idxmax()

@app.callback(
    [
     Output('output-container-date-picker-single', 'children'),
     Output('graph-peso-promedio', 'figure'),  # Output para el gráfico de Peso_Promedio
     Output('graph-costo-directo', 'figure'),  # Output para el gráfico de Costo_Directo
     Output('bar-chart', 'figure'),
     Output('r2-score-display', 'children'),
    ],
    [
     Input('my-date-picker-single', 'date'),
     Input('item-multiple', 'n_clicks'),
     Input('item-simple', 'n_clicks'),
     Input('item-polinomica', 'n_clicks')
    ]
)

def grafico_predicciones(date, n_clicks_multiple, n_clicks_simple, n_clicks_polinomica):
    global df_costo_predicciones

    r2_display = "No R2 Score available"
    fig_peso = go.Figure()
    fig_costo = go.Figure()

    # Inicializar el contador de clics para el modelo polinómico si es None
    if n_clicks_polinomica is None:
        n_clicks_polinomica = 0
    if n_clicks_multiple is None:
        n_clicks_multiple = 0
    if n_clicks_simple is None:
        n_clicks_simple = 0

    # Determinar qué modelo tiene la mayor cantidad de clics y por lo tanto debe ser seleccionado
    max_clicks = max(n_clicks_multiple, n_clicks_simple, n_clicks_polinomica)
    if n_clicks_multiple == max_clicks:
        model_type = 'multiple'
    elif n_clicks_simple == max_clicks:
        model_type = 'simple'
    else:
        model_type = 'polinomica'  # Nuevo caso para manejar la regresión polinómica

    # Asegúrate de que 'last_date' se define aquí, antes de cualquier uso en los bloques de código
    last_date = df['Fecha'].max()

    # Preparación de datos comunes
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Numero de dias'] = (df['Fecha'] - df['Fecha'].min()).dt.days
    df_cleaned = df.dropna(subset=['Peso_Promedio', 'Costo_Directo', 'Biomasa', 'Balanceado', 'Sobrevivencia'])
    features = ['Costo_Directo', 'Biomasa', 'Balanceado', 'Sobrevivencia']
    X = df_cleaned[features]
    y_peso = df_cleaned['Peso_Promedio']
    y_costo = df_cleaned['Costo_Directo_Acumulado']

    if model_type == 'multiple':
        # Regresión lineal múltiple para Peso_Promedio
        multiple_regression_model = LinearRegression()
        multiple_regression_model.fit(X, y)
        last_date = df['Fecha'].max()
        future_dates = pd.date_range(start=last_date, periods=71).tolist()[1:]
        future_df_mult = pd.DataFrame()
        future_df_mult['Fecha'] = future_dates
        future_df_mult['Numero de dias'] = (future_df_mult['Fecha'] - df['Fecha'].min()).dt.days
        for feature in features:
            feature_model = LinearRegression()
            feature_df = df[['Numero de dias', feature]].dropna()
            feature_model.fit(feature_df[['Numero de dias']], feature_df[feature])
            future_df_mult[feature] = feature_model.predict(future_df_mult[['Numero de dias']])
        future_df_mult['Peso_Promedio_Pred'] = multiple_regression_model.predict(future_df_mult[features])
        fig_peso.add_trace(go.Scatter(
            x=df['Fecha'],
            y=y_peso,
            mode='lines',
            name='Valores Reales Peso Promedio',
            line=dict(color='blue'),
            fill='tozeroy',  # Esto rellenará el área bajo la línea hasta el eje x
            fillcolor='rgba(0, 0, 255, 0.2)'  # Un color azul claro con transparencia
        ))
        fig_peso.add_trace(go.Scatter(
            x=future_df_mult['Fecha'],
            y=future_df_mult['Peso_Promedio_Pred'],
            mode='lines',
            name='Predicciones Regresión Lineal Múltiple',
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)',
            ))

        # Regresión lineal múltiple para Costo_Directo
        # Entrenar el modelo de regresión múltiple para Costo_Directo acumulado
        multiple_regression_model_costo = LinearRegression()
        multiple_regression_model_costo.fit(X, y_costo)
        future_df_mult['Costo_Directo_Pred_Acumulado'] = multiple_regression_model_costo.predict(future_df_mult[features])
        
        # Añadir valores reales con relleno azul
        fig_costo.add_trace(go.Scatter(
            x=df['Fecha'],
            y=y_costo,  # Asumiendo que y_costo ya es el costo directo acumulado
            mode='lines',
            name='Valores Reales Costo Directo Acumulado',
            line=dict(color='blue'),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.2)'  # Color azul con transparencia
        ))

        # Añadir predicciones con relleno rojo
        fig_costo.add_trace(go.Scatter(
            x=future_df_mult['Fecha'],
            y=future_df_mult['Costo_Directo_Pred_Acumulado'],  # Asumiendo que estas son las predicciones acumuladas
            mode='lines',
            name='Predicciones Costo Directo Acumulado',
            line=dict(color='red'),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)'  # Color rojo con transparencia
        ))
    # Lógica adicional para el modelo polinómico
    if model_type == 'polinomica':
        polynomial_degree = 2  # Por ejemplo, un polinomio de grado 2

        # Crear las características polinómicas para los datos actuales
        poly_features = PolynomialFeatures(degree=polynomial_degree)
        X_poly = poly_features.fit_transform(df_cleaned[['Numero de dias']])
        
        # Regresión polinómica para Peso_Promedio
        poly_model_peso = LinearRegression()
        poly_model_peso.fit(X_poly, y_peso)
        r2_score_value_poly_peso = r2_score(y_peso, poly_model_peso.predict(X_poly))
        r2_display = f"R2 Score (Polinomic Peso): {r2_score_value_poly_peso:.4f}"
        
        # Predecir para fechas futuras
        future_dates_df = pd.DataFrame({'Fecha': pd.date_range(start=last_date, periods=71, freq='D')})
        future_dates_df['Numero de dias'] = (future_dates_df['Fecha'] - df['Fecha'].min()).dt.days
        future_X_poly = PolynomialFeatures(degree=polynomial_degree).fit_transform(future_dates_df[['Numero de dias']])
        future_dates_df['Peso_Promedio_Pred_Poly'] = poly_model_peso.predict(future_X_poly)
        
        # Añadir traza de valores reales con relleno azul
        fig_peso.add_trace(go.Scatter(
            x=df['Fecha'],
            y=y_peso,
            mode='lines',
            name='Valores Reales Peso Promedio',
            line=dict(color='blue'),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.2)'  # Relleno azul claro con transparencia
        ))

        # Añadir traza de predicciones polinómicas con relleno rojo
        fig_peso.add_trace(go.Scatter(
            x=future_dates_df['Fecha'],
            y=future_dates_df['Peso_Promedio_Pred_Poly'],
            mode='lines',
            name='Predicciones Regresión Polinómica Peso',
            line=dict(color='red'),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)'  # Relleno rojo claro con transparencia
        ))


        # Regresión polinómica para Costo_Directo_Acumulado
        poly_model_costo = LinearRegression()
        poly_model_costo.fit(X_poly, y_costo)
        future_dates_df['Costo_Directo_Pred_Acumulado_Poly'] = poly_model_costo.predict(future_X_poly)
        
        # Añadir traza de valores reales
        # Añadir traza de valores reales con relleno azul
        fig_costo.add_trace(go.Scatter(
            x=df['Fecha'],
            y=y_costo,
            mode='lines',
            name='Valores Reales Costo Directo Acumulado',
            line=dict(color='blue'),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.2)'  # Relleno azul claro con transparencia
        ))
        
        # Añadir traza de predicciones polinómicas con relleno rojo
        fig_costo.add_trace(go.Scatter(
            x=future_dates_df['Fecha'],
            y=future_dates_df['Costo_Directo_Pred_Acumulado_Poly'],
            mode='lines',
            name='Predicciones Regresión Polinómica Costo',
            line=dict(color='red'),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)'  # Relleno rojo claro con transparencia
        ))

    elif model_type == 'simple':
        # Regresión lineal simple para Peso_Promedio
        simple_model_peso = LinearRegression()
        simple_model_peso.fit(df[['Numero de dias']], y_peso)
        r2_score_value_simple_peso = simple_model_peso.score(df[['Numero de dias']], y_peso)
        r2_display = f"R2 Score (Simple Peso): {r2_score_value_simple_peso:.4f}"

        date_range_df = pd.DataFrame({'Fecha': pd.date_range(start=df['Fecha'].min(), periods=len(df) + 70)})
        date_range_df['Numero de dias'] = (date_range_df['Fecha'] - df['Fecha'].min()).dt.days
        date_range_df['Predicciones_Simple_Peso'] = simple_model_peso.predict(date_range_df[['Numero de dias']])
        # Añadir los valores reales con línea y relleno azules
        # Añadir los valores reales con línea y relleno azules
        # Añadir los valores reales con línea y relleno azules
        fig_peso.add_trace(go.Scatter(
            x=df['Fecha'],
            y=y_peso,
            mode='lines',
            name='Valores Reales Peso Promedio',
            line=dict(color='blue'),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.2)'  # Relleno azul con transparencia
        ))

        # Asegúrate de que las predicciones comiencen después de los valores reales
        # Encuentra la última fecha de los valores reales
        last_real_date = df['Fecha'].iloc[-1]

        # Filtra el DataFrame de predicciones para que solo incluya fechas después de los valores reales
        future_predictions_df = date_range_df[date_range_df['Fecha'] > last_real_date]

        # Añadir las predicciones con línea y relleno rojos
        fig_peso.add_trace(go.Scatter(
            x=future_predictions_df['Fecha'],
            y=future_predictions_df['Predicciones_Simple_Peso'],
            mode='lines',  # Usar solo 'lines' para no incluir marcadores
            name='Predicciones Regresión Lineal Simple Peso',
            line=dict(color='red', width=2),  # Ajustar el grosor de la línea aquí
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)'  # Relleno rojo con transparencia
        ))

        # Regresión lineal simple para Costo_Directo
        simple_model_costo = LinearRegression()
        simple_model_costo.fit(df[['Numero de dias']], y_costo)
        r2_score_value_simple_costo = simple_model_costo.score(df[['Numero de dias']], y_costo)
        date_range_df['Predicciones_Simple_Costo'] = simple_model_costo.predict(date_range_df[['Numero de dias']])
        # Real values with blue line and fill
        # Real values with blue line and fill
        fig_costo.add_trace(go.Scatter(
            x=df['Fecha'],
            y=df['Costo_Directo_Acumulado'],  # Assuming 'Costo_Directo_Acumulado' is a column with the accumulated cost
            mode='lines',
            name='Valores Reales Costo Directo Acumulado',
            line=dict(color='blue'),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.2)'  # Blue fill with transparency
        ))

        # Separate the real and predicted values into two dataframes, if not already separated
        # This assumes that `date_range_df` has future dates that do not overlap with `df['Fecha']`
        # Adjust the start of the predictions to be right after the last real date
        start_of_predictions = df['Fecha'].iloc[-1] + pd.Timedelta(days=1)
        predictions_df = date_range_df[date_range_df['Fecha'] >= start_of_predictions]

        # Predictions with red line and fill
        fig_costo.add_trace(go.Scatter(
            x=predictions_df['Fecha'],
            y=predictions_df['Predicciones_Simple_Costo'],
            mode='lines',
            name='Predicciones Regresión Lineal Simple Costo',
            line=dict(color='red'),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)'  # Red fill with transparency
        ))

    # Generación de gráfico de pie
    

    # Generación de gráfico de barras
    bar_chart = px.bar(df_weekly, x='Fecha', y='Peso_Promedio', title='Promedio de Peso_Promedio Semanal')
    bar_chart.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    bar_chart.update_traces(marker=dict(opacity=0.3, line=dict(color='black', width=1.0)))

    # Actualizar el layout de fig_peso y fig_costo
    fig_peso.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig_costo.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    return date, fig_peso, fig_costo, bar_chart, 'Precisión: ' + r2_display

@app.callback(
    Output("model-type", "label"),
    [Input("item-multiple", "n_clicks"), Input("item-simple", "n_clicks")]
)
def update_model_options(n_clicks_multiple, n_clicks_simple):

    # Si uno de los valores es None, asignarle 0
    n_clicks_multiple = 0 if n_clicks_multiple is None else n_clicks_multiple
    n_clicks_simple = 0 if n_clicks_simple is None else n_clicks_simple

    # Luego, decide qué modelo se ha seleccionado
    if n_clicks_multiple > n_clicks_simple:
        return 'Regresión lineal múltiple'
    else:
        return 'Regresión lineal simple'


print(date_range_df.columns)

if __name__ == '__main__':
    app.run_server(debug=True)
