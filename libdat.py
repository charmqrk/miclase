import numpy as np
import pandas as pd
import random
from google.colab import drive
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm, chi2
from collections import Counter
from scipy import stats
from statsmodels.stats.anova import anova_lm
from scipy.stats import shapiro, norm, expon, t, uniform
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import seaborn as sns
import math


##
################################################
##

class AnalisisDescriptivo:
    def __init__(self, datos):
        self.datos = np.array(datos)

    def calculo_de_media(self):
        media = np.mean(self.datos)
        return media

    def calculo_de_mediana(self):
        mediana = np.median(self.datos)
        return mediana

    def calculo_de_desvio_estandar(self):
        desvio = np.std(self.datos)
        return desvio

    def calculo_de_cuartiles(self):
        q1 = np.percentile(self.datos, 25)
        q2 = np.percentile(self.datos, 50)
        q3 = np.percentile(self.datos, 75)
        return [q1, q2, q3]

    def resumen_numerico(self):
        res_num = {
        'Media': self.calculo_de_media(),
        'Mediana': self.calculo_de_mediana(),
        'Desvio': self.calculo_de_desvio_estandar(),
        'Cuartiles': self.calculo_de_cuartiles(),
        'Mínimo': min(self.datos),
        'Máximo': max(self.datos)
        }
        return res_num

    def kernel_gaussiano(self, u):
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

    def kernel_uniforme(self, u):
        return 1 if -0.5 <= u <= 0.5 else 0

    def kernel_cuadratico(self, u):
        return (3/4) * (1 - u**2) if -1 <= u <= 1 else 0

    def densidad_nucleo(self, h, kernel, x):
        """Estimación de densidad con kernel especificado."""
        n = len(self.datos)
        density = np.zeros_like(x)

        for j, val in enumerate(x):
            total = 0
            for dato in self.datos:
                u = (val - dato) / h
                if kernel == 'uniforme':
                    total += self.kernel_uniforme(u)
                elif kernel == 'gaussiano':
                    total += self.kernel_gaussiano(u)
                elif kernel == 'cuadratico':
                    total += self.kernel_cuadratico(u)

            density[j] = total / (n * h)

        return density

    def evalua_histograma(self, h, x):
        """Evalúa densidad del histograma en puntos x."""
        bins = np.arange(min(self.datos) - h/2, max(self.datos) + h, h)
        histograma = np.zeros(len(bins)-1)

        for dato in self.datos:
            for j in range(len(bins)-1):
                if bins[j] <= dato < bins[j+1]:
                    histograma[j] += 1
                    break

        freq_rel = histograma / len(self.datos)
        densidad = freq_rel / h

        estim = np.zeros(len(x))
        for idx, val in enumerate(x):
            for j in range(len(bins)-1):
                if bins[j] <= val < bins[j+1]:
                    estim[idx] = densidad[j]
                    break

        return estim

##
################################################
##

class Estimacion:
    def __init__(self, datos):
        """Clase para estimación de densidades."""
        self.datos = np.array(datos)

    def kernel_gaussiano(self, u):
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

    def kernel_uniforme(self, u):
        return 1 if -0.5 <= u <= 0.5 else 0

    def kernel_cuadratico(self, u):
        return (3/4) * (1 - u**2) if -1 <= u <= 1 else 0

    def densidad_nucleo(self, h, kernel, x):
        """Estimación de densidad con kernel especificado."""
        n = len(self.datos)
        density = np.zeros_like(x)

        for j, val in enumerate(x):
            total = 0
            for dato in self.datos:
                u = (val - dato) / h
                if kernel == 'uniforme':
                    total += self.kernel_uniforme(u)
                elif kernel == 'gaussiano':
                    total += self.kernel_gaussiano(u)
                elif kernel == 'cuadratico':
                    total += self.kernel_cuadratico(u)

            density[j] = total / (n * h)

        return density

    def evalua_histograma(self, h, x):
        """Evalúa densidad del histograma en puntos x."""
        bins = np.arange(min(self.datos) - h/2, max(self.datos) + h, h)
        histograma = np.zeros(len(bins)-1)

        for dato in self.datos:
            for j in range(len(bins)-1):
                if bins[j] <= dato < bins[j+1]:
                    histograma[j] += 1
                    break

        freq_rel = histograma / len(self.datos)
        densidad = freq_rel / h

        estim = np.zeros(len(x))
        for idx, val in enumerate(x):
            for j in range(len(bins)-1):
                if bins[j] <= val < bins[j+1]:
                    estim[idx] = densidad[j]
                    break

        return estim

    def miqqplot(self, datos = None):
      if datos is not None:
        self.datos = np.array(datos)

      data = self.datos
      data_ordenada = np.sort(data)
      media = np.mean(data)
      desv = np.std(data)
      data_ord_s = [(i - media)/desv for i in data_ordenada]
      cuantiles_teoricos = [norm.ppf((i+1)/(len(data)+1)) for i in range(len(data))]
      plt.scatter(cuantiles_teoricos, data_ord_s)
      plt.plot(cuantiles_teoricos, cuantiles_teoricos, color='red')
      plt.xlabel('Cuantiles teóricos')
      plt.ylabel('Cuantiles muestrales')
      plt.show()

##
################################################
##

class GeneradoraDeDatos:
    def __init__(self, n):
        """Clase para generar datos con distribuciones conocidas."""
        self.n = n

    def generar_datos_dist_norm(self, media, desvio):
        return np.random.normal(media, desvio, self.n)

    def pdf_norm(self, x, media, desvio):
        return norm.pdf(x, media, desvio)

    def generar_datos_BS(self):
        u = np.random.uniform(size=self.n)
        y = u.copy()
        ind = np.where(u > 0.5)[0]
        y[ind] = np.random.normal(0, 1, size=len(ind))
        for j in range(5):
            ind = np.where((u > j*0.1) & (u <= (j+1)*0.1))[0]
            y[ind] = np.random.normal(j/2 - 1, 1/10, size=len(ind))
        return y

    def pdf_BS(self, x):
        term1 = (1/2) * norm.pdf(x, 0, 1)
        term2 = (1/10) * sum(norm.pdf(x, j/2 - 1, 1/10) for j in range(5))
        return term1 + term2

    def generar_datos_exp(self, beta):
        return np.random.exponential(beta, self.n)

    def pdf_exp(self, x, beta):
        return expon.pdf(x, scale=beta)

    def generar_datos_chi2(self, gl):
        return np.random.chisquare(gl, self.n)

    def pdf_chi2(self, x, gl):
        return chi2.pdf(x, gl)

##
################################################
##

class QQPlot:
    def __init__(self, datos):
        """Clase para generar y analizar gráficos QQ (quantile-quantile)."""
        self.datos = np.array(datos)

    def graficar_qq_normal(self):
        """Genera un QQ plot comparando los datos con una distribución normal estándar."""
        datos_ordenados = np.sort(self.datos)
        media = np.mean(datos_ordenados)
        desviacion = np.std(datos_ordenados)
        datos_estandarizados = (datos_ordenados - media) / desviacion

        # Calcular cuantiles teóricos de la normal
        n = len(datos_ordenados)
        cuantiles_teoricos = norm.ppf((np.arange(n) + 1) / (n + 1))

        # Graficar
        plt.scatter(cuantiles_teoricos, datos_estandarizados, label='Datos vs Normal')
        plt.plot(cuantiles_teoricos, cuantiles_teoricos, color='r', linestyle='--', label='Línea de identidad')
        plt.xlabel('Cuantiles teóricos (Normal)')
        plt.ylabel('Cuantiles muestrales estandarizados')
        plt.title('QQ Plot vs Distribución Normal')
        plt.legend()
        plt.grid()
        plt.show()

    def graficar_qq_exponencial(self):
        """Genera un QQ plot comparando los datos con una distribución exponencial."""
        datos_ordenados = np.sort(self.datos)
        media_muestral = np.mean(datos_ordenados)
        n = len(datos_ordenados)

        # Calcular cuantiles teóricos de la exponencial
        cuantiles_teoricos = -np.log(1 - (np.arange(n) + 1) / (n + 1)) * media_muestral

        # Graficar
        plt.scatter(cuantiles_teoricos, datos_ordenados, label='Datos vs Exponencial')
        plt.plot(cuantiles_teoricos, cuantiles_teoricos, color='r', linestyle='--', label='Línea de identidad')
        plt.xlabel('Cuantiles teóricos (Exponencial)')
        plt.ylabel('Cuantiles muestrales')
        plt.title('QQ Plot vs Distribución Exponencial')
        plt.legend()
        plt.grid()
        plt.show()

    def graficar_qq_tstudent(self, grados_libertad=2):
        """Genera un QQ plot comparando los datos con una distribución t-Student."""
        datos_ordenados = np.sort(self.datos)
        n = len(datos_ordenados)

        # Estandarizar datos (opcional, para comparar con t-Student estandarizada)
        media = np.mean(datos_ordenados)
        desviacion = np.std(datos_ordenados)
        datos_estandarizados = (datos_ordenados - media) / desviacion

        # Calcular cuantiles teóricos de la t-Student
        cuantiles_teoricos = t.ppf((np.arange(n) + 1) / (n + 1), df=grados_libertad)

        # Graficar
        plt.scatter(cuantiles_teoricos, datos_estandarizados, label=f'Datos vs t-Student (ν={grados_libertad})')
        plt.plot(cuantiles_teoricos, cuantiles_teoricos, color='r', linestyle='--', label='Línea de identidad')
        plt.xlabel('Cuantiles teóricos (t-Student)')
        plt.ylabel('Cuantiles muestrales estandarizados')
        plt.title('QQ Plot vs Distribución t-Student')
        plt.legend()
        plt.grid()
        plt.show()
    def graficar_qq_uniforme(self):
          """Genera un QQ plot comparando los datos con una distribución uniforme(0,1)."""
          datos_ordenados = np.sort(self.datos)
          n = len(datos_ordenados)

          # Calcular cuantiles teóricos de la uniforme
          cuantiles_teoricos = uniform.ppf((np.arange(n) + 1) / (n + 1))

          # Graficar
          plt.scatter(cuantiles_teoricos, datos_ordenados, label='Datos vs Uniforme')
          plt.plot(cuantiles_teoricos, cuantiles_teoricos, color='r', linestyle='--', label='Línea de identidad')
          plt.xlabel('Cuantiles teóricos (Uniforme)')
          plt.ylabel('Cuantiles muestrales')
          plt.title('QQ Plot vs Distribución Uniforme')
          plt.legend()
          plt.grid()
          plt.show()

##
################################################
##

def cargar_csv_desde_drive(ruta_drive, separador=',', mostrar_info=True):
    """
    Carga un archivo CSV desde Google Drive y lo convierte en un DataFrame.

    Parámetros:
    -----------
    ruta_drive : str
        Ruta del archivo en Google Drive (ej: '/content/drive/MyDrive/datos.csv')
    separador : str, opcional
        Separador de columnas en el CSV (por defecto ',')
    mostrar_info : bool, opcional
        Si True, muestra información básica del DataFrame (por defecto True)

    Retorna:
    --------
    tuple: (df, info_variables)
        df: DataFrame de pandas con los datos
        info_variables: Diccionario con información de las variables
    """

    # Montar Google Drive (si no está montado)
    try:
        drive.mount('/content/drive')
    except:
        pass  # Si ya está montado, continua

    # Cargar el archivo CSV
    try:
        df = pd.read_csv(ruta_drive, sep=separador)
    except Exception as e:
        raise ValueError(f"Error al cargar el archivo: {str(e)}")

    # Crear diccionario de información de variables
    info_variables = {
        'n_variables': len(df.columns),
        'n_observaciones': len(df),
        'nombres_variables': list(df.columns),
        'tipos_datos': df.dtypes.to_dict(),
        'variables_numericas': df.select_dtypes(include=['number']).columns.tolist(),
        'variables_categoricas': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'variables_fecha': df.select_dtypes(include=['datetime']).columns.tolist()
    }

    # Mostrar información básica si se solicita
    if mostrar_info:
        display(df.head())

    return df, info_variables

##
################################################
##

class prepararDataframe:
    """
    Clase para preparar y transformar DataFrames de pandas con operaciones comunes en preprocesamiento de datos.

    Esta clase proporciona métodos para eliminar columnas, mapear valores y crear variables dummy.
    También incluye un método para devolver una copia del DataFrame transformado.

    Notar que esta clase no modifica el DataFrame original.

    Atributos:
        df (pd.DataFrame): DataFrame original proporcionado por el usuario
        df_preparado (pd.DataFrame): Copia del DataFrame que se modifica con los métodos de la clase

    Métodos:
        eliminar_variables: Elimina columnas especificadas del DataFrame
        mappear_valores: Transforma valores en una columna según un diccionario de mapeo
        preparar_dummies: Crea variables dummy con control sobre la categoría de referencia
    """

    def __init__(self, df):
        """
        Inicializa la clase con el DataFrame a transformar.

        Args:
            df (pd.DataFrame): DataFrame de pandas que se desea preparar/transformar
        """
        self.df = df
        self.df_preparado = df.copy()

    def eliminar_variables(self, variables_a_eliminar):
        """
        Elimina columnas especificadas del DataFrame.

        Args:
            variables_a_eliminar (str or list): Nombre(s) de la(s) columna(s) a eliminar

        Returns:
            None: Modifica el DataFrame interno directamente

        Ejemplo:
            >>> preparador.eliminar_variables('columna_no_necesaria')
            >>> preparador.eliminar_variables(['col1', 'col2', 'col3'])
        """
        self.df_preparado = self.df_preparado.drop(columns=variables_a_eliminar)

    def mappear_valores(self, diccionario_mapeo, columna_a_mapear):
        """
        Transforma los valores de una columna según un diccionario de mapeo.

        Args:
            diccionario_mapeo (dict): Diccionario con mapeo {valor_original: valor_nuevo}
            columna_a_mapear (str): Nombre de la columna a transformar

        Returns:
            None: Modifica el DataFrame interno directamente

        Ejemplo:
            >>> mapeo = {'Sí': 1, 'No': 0}
            >>> preparador.mappear_valores(mapeo, 'columna_respuesta')
        """
        self.df_preparado[columna_a_mapear] = self.df_preparado[columna_a_mapear].map(diccionario_mapeo)

    def preparar_dummies(self, variable_categorica, valor_objetivo=None, prefix=None, prefix_sep='_', mantener_original=False):
        """
        Crea variables dummy para una columna categórica con control avanzado sobre la categoría de referencia.

        Args:
            variable_categorica (str): Nombre de la columna categórica a convertir
            valor_objetivo (str, optional): Valor categórico que será la categoría de referencia (la que se droppea).
                                          Si es None, se usa el primer valor alfabéticamente.
            prefix (str, optional): Prefijo para los nombres de las nuevas columnas.
                                   Si es None, usa el nombre de la columna original.
            prefix_sep (str, optional): Separador entre el prefijo y el valor categórico. Default='_'
            mantener_original (bool, optional): Si True, mantiene la columna original además de las dummies. Default=False

        Returns:
            None: Modifica el DataFrame interno directamente

        Raises:
            ValueError: Si el valor_objetivo no existe en la columna especificada

        Ejemplo:
            >>> # Crea dummies para 'Nivel_Educativo' con 'Secundaria' como referencia
            >>> preparador.preparar_dummies('Nivel_Educativo', valor_objetivo='Secundaria')

            >>> # Crea dummies con prefijo personalizado
            >>> preparador.preparar_dummies('Región', prefix='reg', prefix_sep='-')
        """
        # Configuración de prefijos
        if prefix is None:
            prefix = variable_categorica

        # Verificar que la columna existe
        if variable_categorica not in self.df_preparado.columns:
            raise ValueError(f"La columna '{variable_categorica}' no existe en el DataFrame")

        # Verificar que el valor_objetivo existe si fue especificado
        if valor_objetivo is not None:
            valores_unicos = self.df_preparado[variable_categorica].unique()
            if valor_objetivo not in valores_unicos:
                raise ValueError(f"El valor '{valor_objetivo}' no existe en la columna '{variable_categorica}'. Valores disponibles: {list(valores_unicos)}")

        # Crear dummies temporales sin droppear ninguna categoría
        dummies = pd.get_dummies(
            self.df_preparado[variable_categorica],
            prefix=prefix,
            prefix_sep=prefix_sep,
            dtype=int  # Usar 0/1 en lugar de True/False
        )

        # Eliminar la columna del valor objetivo si fue especificado
        if valor_objetivo is not None:
            columna_a_eliminar = f"{prefix}{prefix_sep}{valor_objetivo}"
            if columna_a_eliminar in dummies.columns:
                dummies = dummies.drop(columns=[columna_a_eliminar])

        # Eliminar la columna original si no se solicita mantenerla
        if not mantener_original:
            self.df_preparado = self.df_preparado.drop(columns=[variable_categorica])

        # Unir las dummies al DataFrame
        self.df_preparado = pd.concat([self.df_preparado, dummies], axis=1)

    def devolver_df(self):
        """
        Devuelve el DataFrame preparado.

        Returns:
            pd.DataFrame: Copia del DataFrame modificado con los métodos de la clase.
        """
        return self.df_preparado

##
################################################
##

class Regresion:
  def __init__(self, df, variable_respuesta):
      self.df = df.copy()
      self.variable_respuesta = variable_respuesta
      self.variables_predictoras = [col for col in self.df.columns if col != variable_respuesta]
      self.X = df[self.variables_predictoras]
      self.y = df[self.variable_respuesta]
      self.df_train = None
      self.df_test = None
      self.X_test = None
      self.y_test = None
      self.X_train = None
      self.y_train = None
      self._modelo = None
      self._modelo_train = None
      self.resultados = None
      self.resultados_train = None

  def dividir_datos_train_test(self, test_size=0.2, random_state=10, tipo = "ttt", indice_div = None):
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    if tipo == "ttt":
        self.df_train, self.df_test = train_test_split(self.df, test_size=test_size, random_state=random_state)
        self.y_test = self.df_test[self.variable_respuesta]
        self.X_test = self.df_test.drop(columns=[self.variable_respuesta])
        self.X_train = self.df_train.drop(columns=[self.variable_respuesta])
        self.y_train = self.df_train[self.variable_respuesta]
    elif tipo == "random":
        random.seed(random_state)
        indices = random.sample(range(len(self.df)), int(len(self.df) * (1 - test_size)))
        self.df_train = self.df.iloc[indices]
        self.df_test = self.df.drop(indices)
        self.y_test = self.df_test[self.variable_respuesta]
        self.X_test = self.df_test.drop(columns=[self.variable_respuesta])
        self.X_train = self.df_train.drop(columns=[self.variable_respuesta])
        self.y_train = self.df_train[self.variable_respuesta]
    else:
        if indice_div is not None:
            self.df_train = self.df.iloc[:indice_div]
            self.df_test = self.df.iloc[indice_div:]
            self.y_test = self.df_test[self.variable_respuesta]
            self.X_test = self.df_test.drop(columns=[self.variable_respuesta])
            self.X_train = self.df_train.drop(columns=[self.variable_respuesta])
            self.y_train = self.df_train[self.variable_respuesta]


  def agregar_interaccion(self, var1, var2, prefijo=None):
    """Agrega un término de interacción entre dos variables."""

    if var1 not in self.variables_predictoras or var2 not in self.variables_predictoras:
        raise ValueError("Ambas variables deben existir en el DataFrame")

    nombre_interaccion = f"{prefijo}_" if prefijo else ""
    nombre_interaccion += f"{var1}_x_{var2}"

    if nombre_interaccion not in self.variables_predictoras:
        self.df[nombre_interaccion] = self.df[var1] * self.df[var2]
        self.variables_predictoras.append(nombre_interaccion)

##
################################################
##

class RegresionLineal(Regresion):
    def __init__(self, df, variable_respuesta):
        super().__init__(df, variable_respuesta)
        self.resultados = None

    def _calcular_coeficientes_simple(self, x, y):
        """Calcula los coeficientes de la recta de regresión."""
        media_x = np.mean(x)
        media_y = np.mean(y)
        b1 = np.sum((x - media_x) * (y - media_y)) / np.sum((x - media_x) ** 2)
        b0 = media_y - b1 * media_x
        return b0, b1

    def ajustar_modelo(self, train = False):
        """Ajusta el modelo de regresión lineal."""
        if train:
            if self.X_train is None:
                raise ValueError("No hay datos de entrenamiento. Llama a dividir_datos_train_test() primero.")
            else:
                # Usar datos de entrenamiento
                X_train = sm.add_constant(self.X_train)
                self._modelo_train = sm.OLS(self.y_train, X_train)
                self.resultados_train = self._modelo_train.fit()
        else:
            # Si no hay partición, usar todos los datos
            X = sm.add_constant(self.X)
            self._modelo = sm.OLS(self.y, X)
            self.resultados = self._modelo.fit()
            return self.resultados

    def resumen_modelo(self):
        """Muestra un resumen del modelo ajustado."""
        if self.resultados is None:
            self.ajustar_modelo()
        return self.resultados.summary()

    def resumen_modelo_train(self):
        """Muestra un resumen del modelo ajustado entrenado."""
        if self.resultados_train is None:
            self.ajustar_modelo(True)
        return self.resultados_train.summary()

    def graficar_dispersion(self):
        """Grafica la dispersión de puntos para cada variable predictora."""

        num_vars = len(self.variables_predictoras)
        cols = min(3, num_vars)
        rows = (num_vars + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axs = np.array([[axs]])
        elif rows == 1 or cols == 1:
            axs = axs.reshape(-1, 1) if rows > 1 else axs.reshape(1, -1)

        for i, var in enumerate(self.variables_predictoras):
            row_idx = i // cols
            col_idx = i % cols

            # Datos para graficar
            x_data = self.X[var]
            y_data = self.y

            axs[row_idx, col_idx].scatter(x_data, y_data, alpha=0.5)
            axs[row_idx, col_idx].set_title(f'{self.variable_respuesta} vs {var}')
            axs[row_idx, col_idx].set_xlabel(var)
            if col_idx == 0:
                axs[row_idx, col_idx].set_ylabel(self.variable_respuesta)

        # Ocultar ejes vacíos si hay más subplots que variables
        for i in range(num_vars, rows*cols):
            row_idx = i // cols
            col_idx = i % cols
            axs[row_idx, col_idx].axis('off')

        plt.tight_layout()
        plt.show()

    def graficar_dispersion_train(self):
        """Grafica la dispersión de puntos para cada variable predictora entrenada."""

        num_vars = len(self.variables_predictoras)
        cols = min(3, num_vars)
        rows = (num_vars + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axs = np.array([[axs]])
        elif rows == 1 or cols == 1:
            axs = axs.reshape(-1, 1) if rows > 1 else axs.reshape(1, -1)

        for i, var in enumerate(self.variables_predictoras):
            row_idx = i // cols
            col_idx = i % cols

            # Datos para graficar
            x_data = self.X_train[var]
            y_data = self.y_train

            axs[row_idx, col_idx].scatter(x_data, y_data, alpha=0.5)
            axs[row_idx, col_idx].set_title(f'{self.variable_respuesta} vs {var}')
            axs[row_idx, col_idx].set_xlabel(var)
            if col_idx == 0:
                axs[row_idx, col_idx].set_ylabel(self.variable_respuesta)

        # Ocultar ejes vacíos si hay más subplots que variables
        for i in range(num_vars, rows*cols):
            row_idx = i // cols
            col_idx = i % cols
            axs[row_idx, col_idx].axis('off')

        plt.tight_layout()
        plt.show()

    def graficar_recta_regresion(self, variable=None):
        """
        Grafica la recta de regresión para una variable predictora específica.

        Args:
            variable (str): Nombre de la variable predictora a graficar. Si None, usa la primera.
        """
        if self.resultados is None:
            self.ajustar_modelo()

        if variable is None:
            variable = self.variables_predictoras[0]
        elif variable not in self.variables_predictoras:
            raise ValueError(f"La variable {variable} no es una predictora del modelo")

        # Datos para graficar
        x_data = self.X[variable]
        y_data = self.y

        # Valores para la recta de regresión
        x_min, x_max = x_data.min(), x_data.max()
        x_recta = np.linspace(x_min, x_max, 100)

        # Preparamos datos para predecir (solo esta variable)
        datos_recta = pd.DataFrame({variable: x_recta})
        for var in self.variables_predictoras:
            if var != variable:
                datos_recta[var] = x_data.mean()  # Valores medios para otras variables

        # Predecir valores
        y_recta = self.predecir(datos_recta)['prediccion']

        # Graficar
        plt.figure(figsize=(8, 6))
        plt.scatter(x_data, y_data, alpha=0.5, label='Datos observados')
        plt.plot(x_recta, y_recta, color='red', label='Recta de regresión')
        plt.xlabel(variable)
        plt.ylabel(self.variable_respuesta)
        plt.title(f'Regresión lineal: {self.variable_respuesta} ~ {variable}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def graficos_regresion_parcial(self):
        """
        Genera gráficos de regresión parcial para cada variable predictora.
        """
        if self.resultados is None:
            self.ajustar_modelo()

        num_vars = len(self.variable_respuesta)
        cols = min(3, num_vars)
        rows = (num_vars + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axs = np.array([[axs]])
        elif rows == 1 or cols == 1:
            axs = axs.reshape(-1, 1) if rows > 1 else axs.reshape(1, -1)

        for i, var in enumerate(self.variable_respuesta):
            row_idx = i // cols
            col_idx = i % cols

            # Datos para graficar
            x_data = self.X[var]
            residuos = self.resultados.resid

            # Gráfico de regresión parcial
            axs[row_idx, col_idx].scatter(x_data, residuos, alpha=0.5)
            axs[row_idx, col_idx].axhline(y=0, color='r', linestyle='--')
            axs[row_idx, col_idx].set_title(f'Residuos vs {var}')
            axs[row_idx, col_idx].set_xlabel(var)
            if col_idx == 0:
                axs[row_idx, col_idx].set_ylabel('Residuos')

        # Ocultar ejes vacíos
        for i in range(num_vars, rows*cols):
            row_idx = i // cols
            col_idx = i % cols
            axs[row_idx, col_idx].axis('off')

        plt.tight_layout()
        plt.show()

    def graficos_regresion_parcialTrain(self):
        """
        Genera gráficos de regresión parcial para cada variable predictora entrenada.
        """
        if self.resultados_train is None:
            self.ajustar_modelo(True)

        num_vars = len(self.variable_respuesta)
        cols = min(3, num_vars)
        rows = (num_vars + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axs = np.array([[axs]])
        elif rows == 1 or cols == 1:
            axs = axs.reshape(-1, 1) if rows > 1 else axs.reshape(1, -1)

        for i, var in enumerate(self.variable_respuesta):
            row_idx = i // cols
            col_idx = i % cols

            # Datos para graficar
            x_data = self.X_train[var]
            residuos = self.resultados_train.resid

            # Gráfico de regresión parcial
            axs[row_idx, col_idx].scatter(x_data, residuos, alpha=0.5)
            axs[row_idx, col_idx].axhline(y=0, color='r', linestyle='--')
            axs[row_idx, col_idx].set_title(f'Residuos vs {var}')
            axs[row_idx, col_idx].set_xlabel(var)
            if col_idx == 0:
                axs[row_idx, col_idx].set_ylabel('Residuos')

        # Ocultar ejes vacíos
        for i in range(num_vars, rows*cols):
            row_idx = i // cols
            col_idx = i % cols
            axs[row_idx, col_idx].axis('off')

        plt.tight_layout()
        plt.show()

    def coeficiente_correlacion(self, train=False):
        """
        Calcula los coeficientes de correlación entre cada variable predictora y la respuesta.

        Returns:
            pd.Series: Coeficientes de correlación para cada variable predictora
        """

        # Usamos datos de entrenamiento si están disponibles, sino todos los datos
        datos = self.df_train if train else self.df

        # Calculamos correlación para cada variable predictora
        correlaciones = datos[self.variables_predictoras].corrwith(datos[self.variable_respuesta])

        return correlaciones

    def analizar_residuos(self):
        """Realiza el análisis de residuos del modelo."""
        if self.resultados is None:
            self.ajustar_modelo()

        residuos = self.resultados.resid
        predichos = self.resultados.fittedvalues

        plt.figure(figsize=(15, 5))

        # Gráfico de residuos vs predichos
        plt.subplot(1, 2, 1)
        plt.scatter(predichos, residuos)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Valores predichos')
        plt.ylabel('Residuos')
        plt.title('Residuos vs. Valores predichos')

        # QQ plot
        plt.subplot(1, 2, 2)
        qq = QQPlot(residuos)
        qq.graficar_qq_normal()

        plt.tight_layout()
        plt.show()

    def predecir(self, nuevos_datos, intervalo_confianza=False, confianza=0.95):
        """Realiza predicciones con el modelo ajustado."""
        if self.resultados is None:
            self.ajustar_modelo()

        # Validar tipo de datos de entrada
        if isinstance(nuevos_datos, dict):
            nuevos_datos = pd.DataFrame([nuevos_datos])
        elif not isinstance(nuevos_datos, pd.DataFrame):
            raise TypeError("nuevos_datos debe ser un DataFrame o diccionario")

        # Validar columnas
        faltantes = set(self.variables_predictoras) - set(nuevos_datos.columns)
        if faltantes:
            raise ValueError(f"Faltan variables predictoras: {faltantes}")

        # Añadir constante si el modelo la incluye
        if self.resultados.model.k_constant:
            nuevos_datos = sm.add_constant(nuevos_datos, has_constant='add')

        # Realizar predicción
        try:
            pred = self.resultados.get_prediction(nuevos_datos)
            pred_df = pd.DataFrame({'prediccion': pred.predicted_mean})

            if intervalo_confianza:
                ic = pred.conf_int(alpha=1-confianza)
                pred_df['ic_inf'] = ic[:, 0]
                pred_df['ic_sup'] = ic[:, 1]

            return pred_df
        except Exception as e:
            raise ValueError(f"Error en predicción: {str(e)}")

    def predecirTrain(self, nuevos_datos, intervalo_confianza=False, confianza=0.95):
        """Realiza predicciones con el modelo ajustado entrenado."""
        if self.resultados_train is None:
            self.ajustar_modelo(True)

        # Validar tipo de datos de entrada
        if isinstance(nuevos_datos, dict):
            nuevos_datos = pd.DataFrame([nuevos_datos])
        elif not isinstance(nuevos_datos, pd.DataFrame):
            raise TypeError("nuevos_datos debe ser un DataFrame o diccionario")

        # Validar columnas
        faltantes = set(self.variables_predictoras) - set(nuevos_datos.columns)
        if faltantes:
            raise ValueError(f"Faltan variables predictoras: {faltantes}")

        # Añadir constante si el modelo la incluye
        if self.resultados_train.model.k_constant:
            nuevos_datos = sm.add_constant(nuevos_datos, has_constant='add')

        # Realizar predicción
        try:
            pred = self.resultados_train.get_prediction(nuevos_datos)
            pred_df = pd.DataFrame({'prediccion': pred.predicted_mean})

            if intervalo_confianza:
                ic = pred.conf_int(alpha=1-confianza)
                pred_df['ic_inf'] = ic[:, 0]
                pred_df['ic_sup'] = ic[:, 1]

            return pred_df
        except Exception as e:
            raise ValueError(f"Error en predicción: {str(e)}")

    def intervalo_confianza(self, variable, confianza=0.95):
        """Calcula intervalo de confianza para un coeficiente."""
        if self.resultados is None:
            self.ajustar_modelo()

        if variable == 'intercepto':
            idx = 0
        elif variable in self.variables_predictoras:
            idx = self.variables_predictoras.index(variable) + 1
        else:
            raise ValueError(f"Variable {variable} no encontrada en predictores")

        alpha = 1 - confianza
        t_val = stats.t.ppf(1 - alpha/2, self.resultados.df_resid)
        coef = self.resultados.params[idx]
        margen_error = t_val * self.resultados.bse[idx]

        ic_inf = coef - margen_error
        ic_sup = coef + margen_error

        print(f"Intervalo de confianza del {confianza*100:.0f}% para {variable}: ({ic_inf:.4f}, {ic_sup:.4f})")
        return (ic_inf, ic_sup)

    def test_hipotesis(self, variable=None, valor_h0=0, alpha=0.05):
        """Realiza test de hipótesis para coeficientes."""
        if self.resultados is None:
            self.ajustar_modelo()

        if variable is None:
            # Test para todos los coeficientes
            print(self.resultados.summary())
            return

        if variable == 'intercepto':
            idx = 0
        elif variable in self.variables_predictoras:
            idx = self.variables_predictoras.index(variable) + 1
        else:
            raise ValueError(f"Variable {variable} no encontrada en predictores")

        coef = self.resultados.params[idx]
        se = self.resultados.bse[idx]
        t_obs = (coef - valor_h0) / se
        p_valor = 2 * (1 - stats.t.cdf(abs(t_obs), self.resultados.df_resid))

        print(f"Test para H0: {variable} = {valor_h0} vs H1: {variable} ≠ {valor_h0}")
        print(f"Estadístico t observado: {t_obs:.4f}")
        print(f"p-valor: {p_valor:.4f}")

        if p_valor < alpha:
            print(f"Conclusión: Rechazamos H0 (α={alpha})")
        else:
            print(f"Conclusión: No rechazamos H0 (α={alpha})")

    def metricas_modelo(self):
        """Calcula métricas clave del modelo."""
        if self.resultados is None:
            self.ajustar_modelo()

        return {
            'R_cuadrado': self.resultados.rsquared,
            'R_cuadrado_ajustado': self.resultados.rsquared_adj,
            'AIC': self.resultados.aic,
            'BIC': self.resultados.bic
        }
##
################################################
##

class RegresionLogistica(Regresion):
    def __init__(self, df, variable_respuesta, random_state=None):
        super().__init__(df, variable_respuesta)
        self._random_state = random_state
        self._umbral = 0.5
        self.resultados = None
        self.resultados_train = None

    def umbral(self):
        """Getter para el umbral de clasificación."""
        return self._umbral

    def umbral(self, value):
        """Setter para el umbral de clasificación con validación."""
        if not 0 <= value <= 1:
            raise ValueError("El umbral debe estar entre 0 y 1")
        self._umbral = value

    def ajustar_modelo(self, train = False):
        """Ajusta el modelo de regresión logística."""
        if train:
            # Changed _df_train to df_train
            if self.df_train is None:
                raise ValueError("No hay datos de entrenamiento. Llama a dividir_datos_train_test() primero.")
            else:
                # Usar datos de entrenamiento
                X_train = sm.add_constant(self.X_train)
                self._modelo_train = sm.Logit(self.y_train, X_train)
                self.resultados_train = self._modelo_train.fit()
                return self.resultados_train
        else:
            # Si no hay partición, usar todos los datos
            X = sm.add_constant(self.X)
            self._modelo = sm.Logit(self.y, X)
            self.resultados = self._modelo.fit()
            return self.resultados

    def resumen_modelo(self):
        """Muestra un resumen del modelo ajustado."""
        if self.resultados is None:
            if self._modelo is None:
                self.ajustar_modelo()
        return self.resultados.summary()

    def resumen_modelo_train(self):
        """Muestra un resumen del modelo ajustado entrenado."""
        if self.resultados_train is None:
            if self._modelo_train is None:
                self.ajustar_modelo(True)
        return self.resultados_train.summary()


    def predecir(self, nuevos_datos, probabilidades=False):
        """Realiza predicciones con el modelo ajustado."""
        if self.resultados is None:
            if self._modelo is None:
                self.ajustar_modelo()

        # Preparar nuevos datos
        if isinstance(nuevos_datos, dict):
            nuevos_datos = pd.DataFrame([nuevos_datos])

        # Asegurar que tenemos todas las columnas necesarias
        nuevos_datos = nuevos_datos.copy()
        for var in self.variables_predictoras:
            if var not in nuevos_datos.columns:
                 nuevos_datos[var] = 0

        if self._modelo is not None:
             model_cols = list(self._modelo.exog_names)
             if 'const' in model_cols:
                 model_cols.remove('const')
             nuevos_datos = nuevos_datos[self.variables_predictoras]
             nuevos_datos = sm.add_constant(nuevos_datos, has_constant='add')
             nuevos_datos = nuevos_datos[self._modelo.exog_names]
        else:
             nuevos_datos = sm.add_constant(nuevos_datos, has_constant='add')


        proba = self.resultados.predict(nuevos_datos)

        if probabilidades:
            return proba
        else:
            return (proba > self._umbral).astype(int).rename('prediccion').to_frame()


    def predecirTrain(self, nuevos_datos, probabilidades=False):
        """Realiza predicciones con el modelo ajustado entrenado."""
        if self.resultados_train is None:
            # Check if model has been trained on training data
            if self._modelo_train is None:
                self.ajustar_modelo(True)

        # Preparar nuevos datos
        if isinstance(nuevos_datos, dict):
            nuevos_datos = pd.DataFrame([nuevos_datos])

        nuevos_datos = nuevos_datos.copy()

        for var in self.X_train.columns:
            if var not in nuevos_datos.columns:
                 nuevos_datos[var] = 0

        nuevos_datos = nuevos_datos[self.X_train.columns]

        nuevos_datos = sm.add_constant(nuevos_datos, has_constant='add')

        if self._modelo_train is not None:
             nuevos_datos = nuevos_datos[self._modelo_train.exog_names]


        # Calcular probabilidades
        proba = self.resultados_train.predict(nuevos_datos)

        if probabilidades:
            return proba
        else:
            # Ensure prediction output matches input index
            return (proba > self._umbral).astype(int).rename('prediccion').to_frame()

    def evaluar_modelo(self):
        """Evalúa el modelo en los datos de prueba."""
        if self.df_test is None:
            raise ValueError("No hay datos de prueba. Llama a dividir_datos_train_test() primero.")

        if self.resultados_train is None:
            self.ajustar_modelo(True)


        if not hasattr(self, 'resultados_train') or self.resultados_train is None:
             raise ValueError("Debes ajustar el modelo primero con train=True")

        missing_cols_in_test = set(self.X_train.columns) - set(self.X_test.columns)
        for c in missing_cols_in_test:
             self.X_test[c] = 0 # Add missing columns with default value (e.g., 0) - careful with interpretation

        self.X_test = self.X_test[self.X_train.columns]


        y_pred = self.predecirTrain(self.X_test)['prediccion'] # Get the 'prediccion' Series

        # Matriz de confusión
        vp = np.sum((y_pred == 1) & (self.y_test == 1))
        fp = np.sum((y_pred == 1) & (self.y_test == 0))
        fn = np.sum((y_pred == 0) & (self.y_test == 1))
        vn = np.sum((y_pred == 0) & (self.y_test == 0))

        # Métricas
        total = vp + fp + fn + vn
        sensibilidad = vp / (vp + fn) if (vp + fn) > 0 else 0
        especificidad = vn / (vn + fp) if (vn + fp) > 0 else 0
        exactitud = (vp + vn) / total if total > 0 else 0
        error = 1 - exactitud

        return {
            'matriz_confusion': pd.DataFrame({
                'Real=1': [vp, fn,  vp + fn],
                'Real=0': [fp, vn, fp + vn],
                'CuantPred': [vp + fp, fn + vn, total]
            }, index=['Pred=1', 'Pred=0', 'CuantReal']),
            'metricas': {
                'sensibilidad': sensibilidad,
                'especificidad': especificidad,
                'exactitud': exactitud,
                'error_clasificacion': error
            }
        }


    def curva_roc(self):
        """
        Calcula los datos para la curva ROC y el Área bajo la Curva (AUC).

        Returns:
            dict: 'fpr', 'tpr' (listas para graficar) y 'auc' (valor numérico)
        """
        if self.df_test is None: # Changed _df_test to df_test
            raise ValueError("Se requieren datos de prueba. Use dividir_datos_train_test() primero.")

        if self.resultados_train is None:
            self.ajustar_modelo(train=True)

        # Check if the fitted model object exists
        if not hasattr(self, 'resultados_train') or self.resultados_train is None:
             raise ValueError("Debes ajustar el modelo primero con train=True")


        # Probabilidades predichas en los datos de prueba
        # Pass X_test directly to predecirTrain
        probas = self.predecirTrain(self.X_test, probabilidades=True)

        # Ensure y_test is a 1D array or Series for roc_curve
        y_true = self.y_test.values if isinstance(self.y_test, pd.Series) else self.y_test

        # Calcular la curva ROC
        fpr, tpr, thresholds = roc_curve(y_true, probas)

        # Calcular el AUC
        roc_auc = auc(fpr, tpr)

        return {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'thresholds': thresholds
        }


    def graficar_curva_roc(self):
        """Grafica la curva ROC."""
        roc_data = self.curva_roc()

        plt.figure()
        plt.plot(roc_data['fpr'], roc_data['tpr'], color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_data["auc"]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos (FPR)')
        plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
        plt.title('Curva Característica Operativa del Receptor (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()


    def encontrar_mejor_umbral(self, metricas='youden', pasos=100):
        """
        Encuentra el umbral óptimo basado en diferentes métricas.

        Args:
            metricas (str): 'f1', 'exactitud', 'sensibilidad', 'especificidad' o 'youden'
            pasos (int): Número de pasos para evaluar entre 0 y 1

        Returns:
            float: Mejor umbral encontrado
        """
        if self.df_test is None: # Changed _df_test to df_test
            raise ValueError("Se requieren datos de prueba. Use dividir_datos_train_test() primero.")

        if self.resultados_train is None:
            self.ajustar_modelo(True)

        # Check if the fitted model object exists
        if not hasattr(self, 'resultados_train') or self.resultados_train is None:
             raise ValueError("Debes ajustar el modelo primero con train=True")

        # Probabilidades predichas
        # Pass X_test directly to predecirTrain
        probas = self.predecirTrain(self.X_test, probabilidades=True)

        # Evaluar diferentes umbrales
        mejor_valor_metrica = -1 # Use -1 as initial value for finding max
        mejor_umbral = 0.5

        # Use the values of y_test as numpy array for easier comparison
        y_true_vals = self.y_test.values

        for umbral in np.linspace(0, 1, pasos):
            y_pred = (probas > umbral).astype(int)

            vp = np.sum((y_pred == 1) & (y_true_vals == 1))
            fp = np.sum((y_pred == 1) & (y_true_vals == 0))
            fn = np.sum((y_pred == 0) & (y_true_vals == 1))
            vn = np.sum((y_pred == 0) & (y_true_vals == 0))

            sensibilidad = vp / (vp + fn) if (vp + fn) > 0 else 0
            especificidad = vn / (vn + fp) if (vn + fp) > 0 else 0
            exactitud = (vp + vn) / (vp + fp + fn + vn) if (vp + fp + fn + vn) > 0 else 0
            precision = vp / (vp + fp) if (vp + fp) > 0 else 0 # Calculate precision for F1 score
            f1 = 2 * (precision * sensibilidad) / (precision + sensibilidad) if (precision + sensibilidad) > 0 else 0

            # Select metric based on input string
            if metricas == 'f1':
                valor_metrica = f1
            elif metricas == 'exactitud':
                valor_metrica = exactitud
            elif metricas == 'sensibilidad':
                valor_metrica = sensibilidad
            elif metricas == 'especificidad':
                valor_metrica = especificidad
            elif metricas == 'youden':
                valor_metrica = sensibilidad + especificidad - 1
            else:
                raise ValueError("Métrica no soportada. Use 'f1', 'exactitud', 'sensibilidad', 'especificidad' o 'youden'.")

            # Update best threshold if current metric value is better
            if valor_metrica > mejor_valor_metrica:
                 mejor_valor_metrica = valor_metrica
                 mejor_umbral = umbral


        return mejor_umbral
    def graficar_metricas_vs_umbral(self, pasos=100, figsize=(10, 6)):
        """
        Grafica sensibilidad y especificidad en función del umbral de decisión.

        Args:
            pasos (int): Número de puntos a evaluar entre 0 y 1
            figsize (tuple): Tamaño de la figura
        """
        umbrales = np.linspace(0, 1, pasos)
        sensibilidades = []
        especificidades = []

        # Guardar el umbral original para restaurarlo después
        umbral_original = self.umbral

        for p in umbrales:
            self.umbral = p
            eval_p = self.evaluar_modelo()
            metricas = eval_p['metricas']
            sensibilidades.append(metricas['sensibilidad'])
            especificidades.append(metricas['especificidad'])

        # Restaurar el umbral original
        self.umbral = umbral_original

        # Obtener el mejor umbral según Youden
        mejor_umbral = self.encontrar_mejor_umbral(metricas='youden')
        idx_optimo = np.argmin(np.abs(umbrales - mejor_umbral))

        # Crear el gráfico
        plt.figure(figsize=figsize)
        plt.plot(umbrales, sensibilidades, label='Sensibilidad (TPR)', linewidth=2)
        plt.plot(umbrales, especificidades, label='Especificidad (TNR)', linewidth=2)

        # Marcar el mejor umbral
        plt.axvline(x=mejor_umbral, color='red', linestyle='--',
                    label=f'Umbral óptimo: {mejor_umbral:.2f}')

        # Configuración del gráfico
        plt.title('Sensibilidad y Especificidad vs Umbral de Decisión', fontsize=14)
        plt.xlabel('Umbral de Probabilidad', fontsize=12)
        plt.ylabel('Valor de la Métrica', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Mostrar el punto óptimo
        plt.scatter(umbrales[idx_optimo], sensibilidades[idx_optimo], color='red', s=100)
        plt.scatter(umbrales[idx_optimo], especificidades[idx_optimo], color='red', s=100)

        # Añadir anotación
        plt.annotate(f'Sens: {sensibilidades[idx_optimo]:.2f}\nEspec: {especificidades[idx_optimo]:.2f}',
                     xy=(mejor_umbral, (sensibilidades[idx_optimo] + especificidades[idx_optimo])/2),
                     xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle='->'))

        plt.tight_layout()
        plt.show()
