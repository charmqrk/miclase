# Mi repositorio y librería de Clases utilizadas durante Ciencia de Datos 3

#### Este repositorio contiene módulos Python desarrollados para el cursado de la materia Ciencia de Datos III de la 
#### Licenciatura en Ciencia de Datos (FIQ-UNL). Incluye herramientas para análisis descriptivo, estimación estadística,
#### generación de datos, regresión lineal y logística, y visualización.

# Descripción - Módulo principal: libdat.py

libdat.py es un módulo completo que reúne todas las funcionalidades utilizadas durante el cursado, 
organizadas en clases especializadas para diferentes tareas de análisis de datos.

## Características

### 1. Análisis Descriptivo
#### Clase AnalisisDescriptivo:

- Cálculo de medidas estadísticas (media, mediana, desvío estándar, cuartiles)

- Estimación de densidad mediante kernels (gaussiano, uniforme, cuadrático)

- Evaluación de histogramas

- Generación de resúmenes numéricos completos

### 2. Estimación Estadística
#### Clase Estimacion:

- Métodos no paramétricos para estimación de densidades

- Generación de QQ plots para evaluar normalidad

- Comparación con distribuciones teóricas

### 3. Generación de Datos
#### Clase GeneradoraDeDatos:

- Generación de datos con distribuciones conocidas (normal, exponencial, chi-cuadrado)

- Creación de mezclas de distribuciones (ejemplo Bart-Simpson)

- Funciones de densidad de probabilidad para cada distribución

### 4. Regresión
#### Clase base Regresion:

- Funcionalidad común para modelos de regresión

- División de datos en train-test

- Manejo de variables predictoras y respuesta

#### Clase RegresionLineal:

- Ajuste de modelos lineales simples y múltiples

- Diagnóstico de modelos (análisis de residuos, QQ plots)

- Predicciones con intervalos de confianza

- Pruebas de hipótesis para coeficientes

- Métricas de evaluación (R², AIC, BIC)

#### Clase RegresionLogistica:

- Modelos logísticos binarios

- Evaluación mediante matriz de confusión y curva ROC

- Optimización del umbral de clasificación

- Métricas de desempeño (sensibilidad, especificidad, exactitud)

### 5. Manejo de Datos Categóricos
#### Clase AnalizadorCategoricos:

- Análisis de proporciones

- Pruebas de bondad de ajuste (χ²)

- Visualización de conteos observados vs esperados

### 6. Utilidades Adicionales
- Funciones para carga de datos desde Google Drive

#### Clase prepararDataframe para preprocesamiento

- Generación de variables dummy y manejo de categorías

## Instalación

pip install numpy pandas matplotlib scipy statsmodels scikit-learn seaborn

## Ejemplos de uso
### Análisis Descriptivo Básico
`````python
from libdat import AnalisisDescriptivo
import numpy as np

# Datos de ejemplo
datos = np.random.normal(10, 2, 100)

# Crear instancia y calcular estadísticas
analisis = AnalisisDescriptivo(datos)
print("Resumen estadístico:")
print(analisis.resumen_numerico())

# Estimación de densidad con kernel gaussiano
x = np.linspace(min(datos), max(datos), 100)
densidad = analisis.densidad_nucleo(h=0.5, kernel='gaussiano', x=x)

# Graficar
import matplotlib.pyplot as plt
plt.plot(x, densidad)
plt.title("Estimación de densidad con kernel gaussiano")
plt.show()

`````
### Regresión Lineal Múltiple
`````python
from libdat import RegresionLineal
import pandas as pd

# Datos de ejemplo
data = {
    'ventas': [200, 230, 180, 250, 300, 280, 310, 190, 220, 240],
    'publicidad': [10, 12, 8, 15, 18, 14, 20, 9, 11, 16],
    'precio': [15, 14, 16, 13, 12, 14, 11, 15, 14, 13]
}
df = pd.DataFrame(data)

# Crear y ajustar modelo
modelo = RegresionLineal(df, 'ventas')
modelo.ajustar_modelo()

# Resultados
print(modelo.resumen_modelo())
modelo.graficar_dispersion()
modelo.analizar_residuos()

# Predecir nuevos valores
nuevos_datos = pd.DataFrame({'publicidad': [13, 17], 'precio': [14, 12]})
predicciones = modelo.predecir(nuevos_datos, intervalo_confianza=True)
print("\nPredicciones:")
print(predicciones)
`````

### Regresión Logística

`````python
from libdat import RegresionLogistica
import pandas as pd

# Datos de ejemplo (clasificación binaria)
data = {
    'aprobado': [1, 0, 1, 0, 1, 1, 0, 1, 1, 0],
    'horas_estudio': [8, 2, 6, 3, 9, 7, 1, 5, 8, 2],
    'asistencia': [90, 40, 80, 50, 95, 85, 30, 70, 92, 45]
}
df = pd.DataFrame(data)

# Crear y ajustar modelo
modelo = RegresionLogistica(df, 'aprobado')
modelo.dividir_datos_train_test(test_size=0.3, random_state=42)
modelo.ajustar_modelo(train=True)

# Evaluar modelo
print(modelo.resumen_modelo_train())
evaluacion = modelo.evaluar_modelo()
print("\nMatriz de confusión:")
print(evaluacion['matriz_confusion'])
print("\nMétricas:")
print(evaluacion['metricas'])

# Curva ROC
modelo.graficar_curva_roc()
`````

### Análisis de Datos Categóricos

`````python
from libdat import AnalizadorCategoricos

# Datos de ejemplo (categorías)
datos = ['A', 'B', 'A', 'C', 'B', 'B', 'A', 'C', 'C', 'B']

# Crear analizador
analizador = AnalizadorCategoricos(datos, categorias=['A', 'B', 'C'])

# Resumen y pruebas
print("Resumen de conteos:")
print(analizador.resumen_conteos())

print("\nPrueba de bondad de ajuste (distribución uniforme):")
resultado_prueba = analizador.prueba_bondad_ajuste()
print(f"Estadístico χ²: {resultado_prueba['estadistico']:.2f}")
print(f"p-valor: {resultado_prueba['p_valor']:.4f}")

# Gráfico
analizador.graficar_conteos()
`````

### Generación de Datos

`````python
from libdat import GeneradoraDeDatos
import matplotlib.pyplot as plt

# Generar datos normales
generador = GeneradoraDeDatos(1000)
datos_normales = generador.generar_datos_dist_norm(media=5, desvio=2)

# Generar mezcla de distribuciones Bart Simpson
datos_mezcla = generador.generar_datos_BS()

# Graficar
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(datos_normales, bins=30, density=True)
plt.title("Datos Normales")

plt.subplot(1, 2, 2)
plt.hist(datos_mezcla, bins=30, density=True)
plt.title("Mezcla de Distribuciones")
plt.show()
`````

### Preparación de Datos

`````python
from libdat import prepararDataframe
import pandas as pd

# Datos de ejemplo con variables categóricas
data = {
    'edad': [25, 30, 35, 40, 45],
    'genero': ['M', 'F', 'F', 'M', 'M'],
    'ingreso': [50000, 60000, 55000, 70000, 80000]
}
df = pd.DataFrame(data)

# Preparar dataframe
preparador = prepararDataframe(df)
preparador.preparar_dummies('genero', valor_objetivo='M')
df_preparado = preparador.devolver_df()

print("DataFrame original:")
print(df)
print("\nDataFrame preparado:")
print(df_preparado)
`````
# Descripción - clases_auxiliares.py

Este módulo contiene implementaciones anteriores de clases de regresión lineal, ahora reemplazadas por
las versiones más completas en libdat.py. Se incluye como referencia histórica y para garantizar 
transparencia en el desarrollo.

## Características

### 1. Regresión Lineal Simple
#### Clase auxRegresionLinealSimple:

- Ajuste de modelos de regresión simple

- Cálculo de intervalos de confianza y predicción

- Pruebas de hipótesis para coeficientes

- Diagnóstico de supuestos (normalidad, homocedasticidad)

### 2. Regresión Lineal Múltiple
####  Clase auxRegresionLinealMultiple:

- Manejo automático de variables categóricas

- Selección de variables paso a paso

- Comparación de modelos mediante ANOVA

- Métricas de evaluación de modelos

## Ejemplo de uso

`````python
from clases_auxiliares import auxRegresionLinealSimple


x = [1, 2, 3, 4, 5]
y = [2.1, 3.9, 6.2, 8.1, 10.2]


modelo = auxRegresionLinealSimple(x, y, "Variable X", "Variable Y")
modelo.graficar()
modelo.test_hipotesis_beta1()
`````
