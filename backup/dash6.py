import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Cargar los datos
df_costo = pd.read_csv('./datos/Pis_13_26.csv')
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
# Iniciar la aplicación Dash
app = dash.Dash(__name__)

# Crear el layout de la aplicación
app.layout = html.Div([
    dcc.Graph(id='prediccion-costo')
])
# Definir el callback para actualizar el gráfico con las predicciones
@app.callback(
    Output('prediccion-costo', 'figure'),
    Input('prediccion-costo', 'relayoutData')
)


def grafico_costo(relayoutData):
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

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)


