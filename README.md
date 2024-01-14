![ACOSUX1](https://github.com/sebaburella/Analisis-de-Datos-de-Camarones/assets/106763237/be579218-dcc6-4b4b-a166-3298444403ae)

#  Analisis de Datos de Camarones
Este proyecto se centra en el desarrollo de una interfaz web en Python que permitirá a los usuarios predecir el peso promedio de la producción de camarones utilizando distintos modelos de predicción. Los modelos de predicción que se implementarán en este proyecto incluyen Regresión Lineal, Regresión Lineal Múltiple y Polinomica.

![ACOSUX2](https://github.com/sebaburella/Analisis-de-Datos-de-Camarones/assets/106763237/31d5194d-0520-45f2-8f5a-28045bfb899a)


#  Dashboard 
En este documento, se proporciona una explicación detallada de cada uno de los gráficos que se encuentran en el panel de control, el cual se visualiza a través de la interfaz web.

# Gráfico de Ganancias Aproximadas con Punto Óptimo de Cosecha
![image](https://github.com/sebaburella/Analisis-de-Datos-de-Camarones/assets/106763237/5cd4d734-8deb-451a-a5ef-692163c9bd72)
•	Variables: Fecha y Ganancias_Aprox
•	Visualización: Este gráfico utiliza una línea para mostrar las "Ganancias Aproximadas" a lo largo del tiempo.
•	Punto Óptimo de Cosecha: Este es un marcador en el gráfico que señala el día en el cual se obtiene el máximo beneficio (precio de venta). Es un indicador crucial para saber cuándo es el momento más rentable para cosechar los camarones.
•	Elementos adicionales:
•	Se añade un área bajo la curva con una opacidad del 20% para dar más contexto visual.
•	Se utilizan anotaciones para marcar el "Punto Óptimo de Cosecha" con una flecha y texto.
Este gráfico es extremadamente útil para la toma de decisiones en la producción de camarones. No solo muestra cómo se espera que evolucionen las ganancias, sino que también indica cuándo sería más rentable cosechar.
La ecuación para calcular las "Ganancias Aproximadas" sería:
Ganancias Aproximadas=(Precio por talla basado en Peso Promedio×Cantidad de camarones)−(Costo Directo+Costo Indirecto)
Donde:
•	Precio por talla basado en Peso Promedio: Este sería un valor que se obtiene en función del peso promedio de los camarones y se multiplica por la cantidad total de camarones para obtener un ingreso bruto.
•	Costo Directo + Costo Indirecto: Estos son los costos asociados con la producción de los camarones, que se restan del ingreso bruto para obtener las ganancias netas.



# Gráfico de Predicciones de Peso Promedio
![image](https://github.com/sebaburella/Analisis-de-Datos-de-Camarones/assets/106763237/f578338a-0014-4ae0-be28-c3095dcb6626)
Función: grafico_predicciones
Este gráfico muestra tanto los datos reales como las predicciones del "Peso Promedio" de los camarones.
•	Variables Utilizadas:
•	Datos reales: Fecha, Peso_Promedio
•	Predicciones: Generadas por un modelo de regresión lineal (ya sea simple o múltiple, según la selección del usuario).
•	Visualización:
•	Línea azul para datos reales.
•	Línea roja para predicciones (ya sea de un modelo de regresión múltiple o simple).
•	Elementos Adicionales:
•	El gráfico incluye un valor de R2 que indica qué tan bien se ajusta el modelo a los datos.
•	Se utiliza un área bajo la curva (relleno) para dar un contexto visual de la fluctuación en las métricas.
•	Personalización: El usuario tiene la opción de seleccionar qué tipo de modelo desea para las predicciones (simple o múltiple), lo que afecta la línea de predicciones mostrada.

# Gráfico de Predicciones de Costo Directo Acumulado
![image](https://github.com/sebaburella/Analisis-de-Datos-de-Camarones/assets/106763237/ac80280e-6e34-4128-a12f-3f24cd090a4e)
Función: grafico_costo
Este gráfico muestra los datos reales y las predicciones del "Costo Directo Acumulado".
•	Variables Utilizadas:
•	Datos reales: Fecha, Costo_Directo_Acumulado
•	Predicciones: Costo_Directo_Acumulado_Predicho, generado por una regresión lineal simple. A este modelo se añade un factor de R2 para calcular la precisión del modelo.
•	Visualización:
•	Línea azul para los datos reales del "Costo Directo Acumulado".
•	Línea roja punteada para las predicciones del "Costo Directo Acumulado".
•	Elementos Adicionales:
•	Al igual que en el gráfico anterior, este gráfico utiliza un área bajo la curva para proporcionar un contexto visual adicional.
Ambos gráficos ofrecen una visión completa y detallada de las métricas clave en la producción de camarones, permitiendo una toma de decisiones más informada. Los modelos de predicción, junto con los valores de R2, ofrecen una evaluación cuantitativa del rendimiento del modelo, lo cual es crucial para evaluar la precisión de las predicciones.


# Grafico de Tipo de Balanceado 
![image](https://github.com/sebaburella/Analisis-de-Datos-de-Camarones/assets/106763237/af96a48b-b562-49ed-acde-499e3b23bf45)
Este gráfico es un tipo de gráfico circular que muestra los datos según el tipo de alimento balanceado consumido por los camarones, y representa el porcentaje de cada tipo de alimento en el consumo total.


# Gráfico de Columnas
![image](https://github.com/sebaburella/Analisis-de-Datos-de-Camarones/assets/106763237/b3d519d6-d7a3-4b99-b188-0a8b9de28d7e)
Este gráfico de columnas muestra dos variables: el Peso Promedio Semanal y los Costos Diarios. Cuando se selecciona la variable ‘Peso Promedio’, el gráfico muestra el Peso Promedio Semanal. De manera similar, al seleccionar la variable ‘Costo Directo Acumulado’, el gráfico muestra los Costos Diarios.

# Tabla de Datos
![image](https://github.com/sebaburella/Analisis-de-Datos-de-Camarones/assets/106763237/4826231f-fe73-4933-a314-d367c815d765)
En esta sección del panel de control, se presenta una tabla que contiene datos generales relevantes. Estos datos incluyen:
•	Fecha: Muestra la fecha correspondiente a los registros.
•	Incremento: Este valor está directamente relacionado con el aumento en el peso promedio del camarón. Indica cuánto ha crecido el camarón en términos de peso.
•	FCA (Factor de Conversión Alimenticia): El FCA es una medida que se utiliza para calcular la eficiencia de la alimentación en la piscina de camarones. Representa la cantidad de alimento balanceado proporcionado en relación con el crecimiento obtenido por los camarones.
•	Biomasa: Aquí se muestra la cantidad total de organismos vivos presentes en la piscina. Es una medida de la población de camarones en la instalación.
•	Biomasa por hectárea: Similar a la biomasa anterior, pero se expresa en términos de la cantidad de organismos vivos por hectárea de terreno.
Estos datos proporcionan una visión general de la situación en la piscina de camarones, incluyendo su crecimiento, eficiencia alimenticia y densidad poblacional por área.
