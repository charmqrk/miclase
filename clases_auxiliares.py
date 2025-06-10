"""
Módulo clases_auxiliares.py

Este módulo contiene clases para el analisis de Regresión
Lineal. Son las funciones ya desuso luego de adaptar las clases
siguiendo las pautas de herencia y polimorfismo propuestas
por la cátedra. Agrego esta libreria recopilando estas funciones
que me han funcionado, para garantizar la transparencia del código
que usé durante todo el cursado.

Dependencias:
- numpy
- pandas
- matplotlib
- scipy.stats
- statsmodels
- sklearn
- seaborn
"""

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
from statsmodels.stats.outliers_influence import variance_inflation_factor

class auxRegresionLinealSimple:
    def __init__(self, x, y, nombre_predictora='Variable X', nombre_respuesta='Variable Y'):
        """
        Inicializa el modelo de regresión lineal simple con arrays/listas de datos.

        Es una clase auxiliar para la regresión lineal simple. Se que funciona, se que no me
        falla.

        Parámetros:
        x (array-like): Array o lista con los valores de la variable predictora
        y (array-like): Array o lista con los valores de la variable respuesta
        nombre_predictora (str): Nombre descriptivo para la variable predictora (opcional)
        nombre_respuesta (str): Nombre descriptivo para la variable respuesta (opcional)
        """
        self.predictora = np.asarray(x)
        self.respuesta = np.asarray(y)
        self.nombre_predictora = nombre_predictora
        self.nombre_respuesta = nombre_respuesta

        # Validación de datos
        if len(self.predictora) != len(self.respuesta):
            raise ValueError("Los arrays x e y deben tener la misma longitud")

        self.b0, self.b1 = self._calcular_coeficientes(self.predictora, self.respuesta)
        self.modelo = self._ajustar_modelo()

    def _ajustar_modelo(self):
        """Ajusta el modelo de regresión usando statsmodels."""
        X = sm.add_constant(self.predictora)
        modelo = sm.OLS(self.respuesta, X).fit()
        return modelo

    def _calcular_coeficientes(self, x, y):
        """Calcula los coeficientes de la recta de regresión."""
        media_x = np.mean(x)
        media_y = np.mean(y)
        b1 = np.sum((x - media_x) * (y - media_y)) / np.sum((x - media_x) ** 2)
        b0 = media_y - b1 * media_x
        return b0, b1

    def valores_recta(self, x=None):
        """Calcula los valores predichos por la recta de regresión."""
        if x is None:
            x = self.predictora
        return self.b0 + self.b1 * np.asarray(x)

    def graficar(self, titulo="Regresión Lineal Simple"):
        """Genera un gráfico de dispersión con la recta de regresión."""
        plt.figure(figsize=(8, 6))
        plt.scatter(self.predictora, self.respuesta, marker="o", facecolors="none", edgecolors="blue", label='Datos')
        plt.plot(self.predictora, self.valores_recta(), color="red", label='Recta de regresión')
        plt.xlabel(self.nombre_predictora)
        plt.ylabel(self.nombre_respuesta)
        plt.title(titulo)
        plt.legend()
        plt.grid(True)
        plt.show()

    def residuos(self):
        """Calcula los residuos del modelo."""
        return self.respuesta - self.valores_recta()

    def estimador_varianza(self):
        """Calcula la estimación de la varianza del error."""
        residuos = self.residuos()
        return np.sum(residuos**2) / (len(residuos) - 2)

    def test_hipotesis_beta1(self, valor_h0=0, alpha=0.05):
        """Realiza test de hipótesis para el coeficiente beta1."""
        t_obs = (self.b1 - valor_h0) / self.modelo.bse[1]
        p_valor = 2 * (1 - stats.t.cdf(abs(t_obs), len(self.predictora)-2))

        print(f"Test para H0: β1 = {valor_h0} vs H1: β1 ≠ {valor_h0}")
        print(f"Estadístico t observado: {t_obs:.4f}")
        print(f"p-valor: {p_valor:.4f}")

        if p_valor < alpha:
            print(f"Conclusión: Rechazamos H0 (α={alpha}). Hay evidencia de que β1 ≠ {valor_h0}")
        else:
            print(f"Conclusión: No rechazamos H0 (α={alpha}). No hay evidencia suficiente para afirmar que β1 ≠ {valor_h0}")

    def intervalo_confianza_beta1(self, confianza=0.95):
        """Calcula intervalo de confianza para el coeficiente beta1."""
        alpha = 1 - confianza
        t_val = stats.t.ppf(1 - alpha/2, len(self.predictora)-2)
        margen_error = t_val * self.modelo.bse[1]

        ic_inf = self.b1 - margen_error
        ic_sup = self.b1 + margen_error

        print(f"Intervalo de confianza del {confianza*100:.0f}% para β1: ({ic_inf:.4f}, {ic_sup:.4f})")
        return (ic_inf, ic_sup)

    def _se_confianza(self, x_0):
        """Calcula el error estándar para el intervalo de confianza de la media."""
        sigma2_est = self.estimador_varianza()
        sum_cuadrados = np.sum((self.predictora - np.mean(self.predictora))**2)

        var_beta1 = sigma2_est / sum_cuadrados
        var_beta0 = sigma2_est * np.sum(self.predictora**2) / (len(self.predictora) * sum_cuadrados)
        cov_01 = -np.mean(self.predictora) * sigma2_est / sum_cuadrados

        se2_est = var_beta0 + (x_0**2) * var_beta1 + 2 * x_0 * cov_01
        return np.sqrt(se2_est)

    def _se_prediccion(self, x_0):
        """Calcula el error estándar para el intervalo de predicción."""
        return np.sqrt(self._se_confianza(x_0)**2 + self.estimador_varianza())

    def intervalo_confianza_mu(self, x_0, confianza=0.95):
        """
        Calcula intervalo de confianza para la media condicional E(Y|X=x_0).

        Parámetros:
        x_0: Valor de la variable predictora para el que se quiere estimar la media
        confianza: Nivel de confianza (0-1)

        Retorna:
        Tupla con los límites inferior y superior del intervalo
        """
        alpha = 1 - confianza
        y_hat = self.b0 + self.b1 * x_0
        t_val = stats.t.ppf(1 - alpha/2, len(self.predictora)-2)
        margen_error = t_val * self._se_confianza(x_0)

        ic_inf = y_hat - margen_error
        ic_sup = y_hat + margen_error
        return (ic_inf, ic_sup)

    def intervalo_prediccion(self, x_0, confianza=0.95):
        """
        Calcula intervalo de predicción para un valor individual Y cuando X=x_0.

        Parámetros:
        x_0: Valor de la variable predictora para el que se quiere predecir Y
        confianza: Nivel de confianza (0-1)

        Retorna:
        Tupla con los límites inferior y superior del intervalo
        """
        alpha = 1 - confianza
        y_hat = self.b0 + self.b1 * x_0
        t_val = stats.t.ppf(1 - alpha/2, len(self.predictora)-2)
        margen_error = t_val * self._se_prediccion(x_0)

        ic_inf = y_hat - margen_error
        ic_sup = y_hat + margen_error
        return (ic_inf, ic_sup)

    def resumen_modelo(self):
        """Muestra un resumen completo del modelo de regresión."""
        print(self.modelo.summary())

    def predecir(self, x_0):
        """
        Predice el valor de Y para un valor dado de X.

        Parámetros:
        x_0: Valor de la variable predictora o array de valores

        Retorna:
        Valor(es) predicho(s) de la variable respuesta
        """
        return self.b0 + self.b1 * np.asarray(x_0)

    def r_cuadrado(self):
        """Retorna el coeficiente de determinación R²."""
        return self.modelo.rsquared

    def mediana_condicional(self, x_0, confianza=0.95):
        """
        Estima la mediana condicional de Y dado X=x_0 con intervalo de confianza
        usando regresión cuantílica (quantile regression).

        Parámetros:
        x_0: Valor de la variable predictora
        confianza: Nivel de confianza (0-1)

        Retorna:
        (mediana, (ic_inf, ic_sup))
        """
        # Ajustar modelo de regresión cuantílica para la mediana (quantile=0.5)
        X = sm.add_constant(self.predictora)
        modelo_qr = sm.QuantReg(self.respuesta, X).fit(q=0.5)

        # Preparar el punto de predicción
        X_new = sm.add_constant([1, x_0])

        # Obtener predicción para la mediana
        mediana = modelo_qr.predict(X_new)[0]

        # Obtener intervalo de confianza
        prediccion = modelo_qr.get_prediction(X_new)
        ic_inf, ic_sup = prediccion.conf_int(alpha=1-confianza)[0]

        return mediana, (ic_inf, ic_sup)

    def verificar_supuestos(self):
        """Verifica los supuestos del modelo de regresión lineal."""
        residuos = self.residuos()

        # Test de normalidad
        stat, p_valor = shapiro(residuos)
        print(f"Test de Shapiro-Wilk para normalidad de residuos:")
        print(f"Estadístico = {stat:.4f}, p-valor = {p_valor:.4f}")

        # Gráfico de residuos vs predichos
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(self.valores_recta(), residuos)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Valores predichos')
        plt.ylabel('Residuos')
        plt.title('Residuos vs. Valores predichos')

        # QQ plot
        plt.subplot(1, 2, 2)
        self._qqplot(residuos)
        plt.title('QQ Plot de residuos')

        plt.tight_layout()
        plt.show()

    def _qqplot(self, data):
        """Genera un QQ plot personalizado."""
        media = np.mean(data)
        desviacion = np.std(data)
        data_s = (data - media) / desviacion
        cuantiles_muestrales = np.sort(data_s)
        n = len(data)
        pp = np.arange(1, (n+1))/(n+1)
        cuantiles_teoricos = norm.ppf(pp)

        plt.scatter(cuantiles_teoricos, cuantiles_muestrales, color='blue', marker='o')
        plt.xlabel('Cuantiles teóricos')
        plt.ylabel('Cuantiles muestrales')
        plt.plot(cuantiles_teoricos, cuantiles_teoricos, linestyle='-', color='red')

class auxRegresionLinealMultiple:
    def __init__(self, dataframe, variable_respuesta, variables_predictoras, categoria_base=None):
        """
        Inicializa el modelo de regresión lineal múltiple generalizado.

        Parámetros:
        -----------
        dataframe : pd.DataFrame
            DataFrame con los datos
        variable_respuesta : str
            Nombre de la variable respuesta/dependiente
        variables_predictoras : list
            Lista de nombres de variables predictoras/independientes
        categoria_base : str, optional
            Categoría base para variables categóricas (default=None)
        """
        self.df = dataframe.copy()
        self.y_name = variable_respuesta
        self.x_names = variables_predictoras.copy() if variables_predictoras else []
        self.categoria_base = categoria_base
        self.modelo = None
        self.resultados = None
        self._preparar_datos()

    def _preparar_datos(self):
        """Prepara los datos para el análisis, manejando variables categóricas y numéricas."""
        # Manejo de variables categóricas
        for var in [col for col in self.x_names if col in self.df.select_dtypes(['object', 'category']).columns]:
            if self.categoria_base is not None:
                categories = [c for c in self.df[var].unique() if c != self.categoria_base]
                categories = [self.categoria_base] + categories
            else:
                categories = self.df[var].unique()

            self.df[var] = pd.Categorical(self.df[var], categories=categories)
            dummies = pd.get_dummies(self.df[var], prefix=var, drop_first=True)
            self.df = pd.concat([self.df.drop(var, axis=1), dummies], axis=1)

            # Actualizar nombres de variables predictoras
            self.x_names.remove(var)
            self.x_names.extend(dummies.columns.tolist())

        # Asegurar que las variables son numéricas
        self.y = self.df[self.y_name].astype(float)
        self.X = self.df[self.x_names].astype(float)
        self.X = sm.add_constant(self.X)

    def agregar_interaccion(self, var1, var2, prefijo=None):
        """
        Agrega una interacción entre dos variables al modelo.

        Parámetros:
        -----------
        var1 : str
            Nombre de la primera variable
        var2 : str
            Nombre de la segunda variable
        prefijo : str, optional
            Prefijo para el nombre de la interacción (default=None)
        """
        if var1 not in self.df.columns or var2 not in self.df.columns:
            raise ValueError("Ambas variables deben existir en el DataFrame")

        nombre_interaccion = f"{prefijo}_" if prefijo else ""
        nombre_interaccion += f"{var1}_x_{var2}"

        if nombre_interaccion not in self.df.columns:
            self.df[nombre_interaccion] = self.df[var1] * self.df[var2]
            self.x_names.append(nombre_interaccion)
            self._preparar_datos()  # Re-preparar datos para incluir la interacción

    def ajustar_modelo(self):
        """Ajusta el modelo de regresión lineal múltiple."""
        self.modelo = sm.OLS(self.y, self.X)
        self.resultados = self.modelo.fit()
        return self.resultados

    def resumen_modelo(self):
        """Muestra un resumen completo del modelo."""
        if self.resultados is None:
            self.ajustar_modelo()
        return self.resultados.summary()

    def graficos_diagnostico(self):
        """Genera gráficos de diagnóstico para evaluar supuestos."""
        if self.resultados is None:
            self.ajustar_modelo()

        residuos = self.resultados.resid
        predichos = self.resultados.fittedvalues

        plt.figure(figsize=(15, 10))

        # Gráfico de residuos vs predichos
        plt.subplot(2, 2, 1)
        plt.scatter(predichos, residuos)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Valores predichos')
        plt.ylabel('Residuos')
        plt.title('Residuos vs. Valores predichos')

        # QQ plot
        plt.subplot(2, 2, 2)
        self._qqplot(residuos)
        plt.title('QQ Plot de residuos')

        # Histograma de residuos
        plt.subplot(2, 2, 3)
        plt.hist(residuos, bins=20, density=True, alpha=0.6)
        x = np.linspace(min(residuos), max(residuos), 100)
        plt.plot(x, norm.pdf(x, np.mean(residuos), np.std(residuos)), 'r-')
        plt.title('Distribución de residuos')

        # Gráfico de escala-localización
        plt.subplot(2, 2, 4)
        residuos_estandarizados = np.sqrt(np.abs(residuos - residuos.mean()) / residuos.std())
        plt.scatter(predichos, residuos_estandarizados)
        plt.xlabel('Valores predichos')
        plt.ylabel('Raíz cuadrada de residuos estandarizados')
        plt.title('Gráfico de escala-localización')

        plt.tight_layout()
        plt.show()

    def _qqplot(self, data):
        """Genera un QQ plot personalizado."""
        media = np.mean(data)
        desviacion = np.std(data)
        data_s = (data - media) / desviacion
        cuantiles_muestrales = np.sort(data_s)
        n = len(data)
        pp = np.arange(1, (n+1))/(n+1)
        cuantiles_teoricos = norm.ppf(pp)

        plt.scatter(cuantiles_teoricos, cuantiles_muestrales, color='blue', marker='o')
        plt.xlabel('Cuantiles teóricos')
        plt.ylabel('Cuantiles muestrales')
        plt.plot(cuantiles_teoricos, cuantiles_teoricos, linestyle='-', color='red')

    def test_normalidad(self):
        """Realiza test de Shapiro-Wilk para normalidad de residuos."""
        if self.resultados is None:
            self.ajustar_modelo()

        stat, p_valor = shapiro(self.resultados.resid)
        print(f"Test de Shapiro-Wilk para normalidad:")
        print(f"Estadístico = {stat:.4f}, p-valor = {p_valor:.4f}")

        if p_valor < 0.05:
            print("Conclusión: Rechazamos la hipótesis de normalidad (α=0.05)")
        else:
            print("Conclusión: No hay evidencia para rechazar la normalidad (α=0.05)")

        return p_valor

    def test_hipotesis(self, alpha=0.05):
        """Realiza tests de hipótesis para cada coeficiente."""
        if self.resultados is None:
            self.ajustar_modelo()

        print(f"Tests de hipótesis para coeficientes (α={alpha}):")
        print("H0: βi = 0 vs H1: βi ≠ 0\n")

        for i, var in enumerate(['Intercept'] + self.x_names):
            p_valor = self.resultados.pvalues[i]
            coef = self.resultados.params[i]

            print(f"Variable: {var}")
            print(f"Coeficiente estimado: {coef:.4f}")
            print(f"p-valor: {p_valor:.4f}")

            if p_valor < alpha:
                print(f"Conclusión: Rechazamos H0 (α={alpha}). Hay evidencia de que {var} es significativa.")
            else:
                print(f"Conclusión: No rechazamos H0 (α={alpha}). No hay evidencia suficiente para afirmar que {var} es significativa.")
            print("-" * 50)

    def comparar_modelos(self, modelo_reducido):
        """
        Compara este modelo (completo) con un modelo reducido usando ANOVA.

        Parámetros:
        -----------
        modelo_reducido : RegresionLinealMultiple
            Modelo reducido a comparar

        Retorna:
        --------
        DataFrame con los resultados de la comparación ANOVA
        """
        if self.resultados is None:
            self.ajustar_modelo()
        if modelo_reducido.resultados is None:
            modelo_reducido.ajustar_modelo()

        anova_res = anova_lm(modelo_reducido.resultados, self.resultados)
        print("Comparación de modelos usando ANOVA:")
        print(anova_res)
        return anova_res

    def predecir(self, nuevos_datos, intervalo_confianza=False, intervalo_prediccion=False, confianza=0.95):
        """
        Realiza predicciones con el modelo ajustado.

        Parámetros:
        -----------
        nuevos_datos : dict o pd.DataFrame
            Datos para predecir
        intervalo_confianza : bool, optional
            Si calcular intervalo de confianza (default=False)
        intervalo_prediccion : bool, optional
            Si calcular intervalo de predicción (default=False)
        confianza : float, optional
            Nivel de confianza (0-1) (default=0.95)

        Retorna:
        --------
        pd.DataFrame con predicciones e intervalos si se solicitaron
        """
        if self.resultados is None:
            self.ajustar_modelo()

        # Preparar nuevos datos
        if isinstance(nuevos_datos, dict):
            nuevos_datos = pd.DataFrame([nuevos_datos])

        # Manejar variables categóricas como en el modelo original
        nuevos_datos = nuevos_datos.copy()
        for var in self.x_names:
            if var not in nuevos_datos.columns:
                # Podría ser una dummy de una variable categórica
                nuevos_datos[var] = 0  # Por defecto 0 (categoría base)

        # Asegurar que tenemos todas las columnas necesarias
        columnas_necesarias = ['const'] + self.x_names
        nuevos_datos = sm.add_constant(nuevos_datos, has_constant='add')

        # Verificar que tenemos todas las columnas necesarias
        for col in columnas_necesarias:
            if col not in nuevos_datos.columns:
                nuevos_datos[col] = 0  # Asignar 0 si la columna no existe

        # Seleccionar solo las columnas necesarias en el orden correcto
        nuevos_datos = nuevos_datos[columnas_necesarias]

        # Realizar predicción
        pred = self.resultados.get_prediction(nuevos_datos)
        pred_df = pd.DataFrame({
            'prediccion': pred.predicted_mean,
        })

        if intervalo_confianza:
            ic = pred.conf_int(alpha=1-confianza)
            pred_df['ic_inf'] = ic[:, 0]
            pred_df['ic_sup'] = ic[:, 1]

        if intervalo_prediccion:
            ic_pred = pred.conf_int(obs=True, alpha=1-confianza)
            pred_df['pred_ic_inf'] = ic_pred[:, 0]
            pred_df['pred_ic_sup'] = ic_pred[:, 1]

        return pred_df

    def graficos_exploratorios(self):
        """Genera gráficos de dispersión de Y vs cada X."""
        num_vars = len(self.x_names)
        cols = min(3, num_vars)
        rows = (num_vars + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axs = np.array([[axs]])
        elif rows == 1 or cols == 1:
            axs = axs.reshape(-1, 1) if rows > 1 else axs.reshape(1, -1)

        for i, var in enumerate(self.x_names):
            row_idx = i // cols
            col_idx = i % cols
            axs[row_idx, col_idx].scatter(self.df[var], self.y, alpha=0.5)
            axs[row_idx, col_idx].set_title(f'{self.y_name} vs {var}')
            axs[row_idx, col_idx].set_xlabel(var)
            if col_idx == 0:
                axs[row_idx, col_idx].set_ylabel(self.y_name)

        # Ocultar ejes vacíos si hay más subplots que variables
        for i in range(num_vars, rows*cols):
            row_idx = i // cols
            col_idx = i % cols
            axs[row_idx, col_idx].axis('off')

        plt.tight_layout()
        plt.show()

    def evaluar_rendimiento(self, datos_test=None):
        """
        Evalúa el rendimiento del modelo en datos de entrenamiento y prueba.

        Parámetros:
        -----------
        datos_test : pd.DataFrame, optional
            Datos de prueba para evaluación (default=None)

        Retorna:
        --------
        dict con métricas de rendimiento
        """
        if self.resultados is None:
            self.ajustar_modelo()

        metricas = {
            'R²': self.resultados.rsquared,
            'R² ajustado': self.resultados.rsquared_adj,
            'AIC': self.resultados.aic,
            'BIC': self.resultados.bic,
            'MSE entrenamiento': np.mean(self.resultados.resid**2)
        }

        if datos_test is not None:
            try:
                pred_test = self.predecir(datos_test)
                y_test = datos_test[self.y_name]
                mse_test = np.mean((pred_test['prediccion'] - y_test)**2)
                metricas['MSE test'] = mse_test
            except Exception as e:
                print(f"Error al evaluar en datos test: {str(e)}")
                metricas['MSE test'] = None

        return metricas

    def rcuadrado(self, ajustado = False):
        """
        Calcula el coeficiente de determinación (R²).
        Parámetros:
        -----------
        ajustado : bool, opcional
            Pregunta si se calcula R² ajustado (False predeterminadamente)

        Retorna:
        --------
        float con el valor de R²
        """
        if self.resultados is None:
            self.ajustar_modelo()

        if ajustado:
            return self.resultados.rsquared_adj
        else:
            return self.resultados.rsquared

    def seleccion_variables(self, metodo='adelante', criterio='pvalor', umbral=0.05):
        """
        Realiza selección de variables paso a paso.

        Parámetros:
        -----------
        metodo : str, optional
            'adelante', 'atras' o 'stepwise' (default='adelante')
        criterio : str, optional
            'pvalor', 'aic' o 'bic' (default='pvalor')
        umbral : float, optional
            Umbral para criterio de p-valor (default=0.05)

        Retorna:
        --------
        Lista con las variables seleccionadas
        """
        variables_incluidas = []
        variables_restantes = self.x_names.copy()

        if metodo == 'adelante':
            while variables_restantes:
                mejores_resultados = []
                for var in variables_restantes:
                    vars_temporales = variables_incluidas + [var]
                    modelo_temp = RegresionLinealMultiple(self.df, self.y_name, vars_temporales)
                    resultados = modelo_temp.ajustar_modelo()

                    if criterio == 'pvalor':
                        p_valores = resultados.pvalues[1:]  # Excluir intercepto
                        mejor_p = min(p_valores)
                        mejores_resultados.append((mejor_p, var))
                    elif criterio == 'aic':
                        aic = resultados.aic
                        mejores_resultados.append((aic, var))
                    elif criterio == 'bic':
                        bic = resultados.bic
                        mejores_resultados.append((bic, var))

                if not mejores_resultados:
                    break

                if criterio == 'pvalor':
                    mejor_metrica, mejor_var = min(mejores_resultados)
                    if mejor_metrica < umbral:
                        variables_incluidas.append(mejor_var)
                        variables_restantes.remove(mejor_var)
                    else:
                        break
                else:
                    mejor_metrica, mejor_var = min(mejores_resultados)
                    variables_incluidas.append(mejor_var)
                    variables_restantes.remove(mejor_var)

        # Similar para método 'atras' o 'stepwise'
        self.x_names = variables_incluidas
        self._preparar_datos()
        return variables_incluidas