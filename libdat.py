import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
from google.colab import drive
from scipy import stats
from statsmodels.stats.anova import anova_lm
from scipy.stats import shapiro, norm, expon, t, uniform, chi2
from collections import Counter
import random
import seaborn as sns
import scipy.stats as stats

###
##################
###

def Dummies(dataframe, columna, variableIntercpt):
    dataframe[columna].Categorical(variableIntercpt)
    dataframe = dataframe.getDummies(columna,dropfirst = True)
    return dataframe


class AnalisisDescriptivo:
    def __init__(self, datos):
        self.datos = np.array(datos)

    def calculo_de_media(self):
        ## Completar
        media = np.mean(self.datos)
        return media

    def calculo_de_mediana(self):
        ## Completar
        mediana = np.median(self.datos)
        return mediana

    def calculo_de_desvio_estandar(self):
        ## Completar
        desvio = np.std(self.datos)
        return desvio

    def calculo_de_cuartiles(self):
        ## Completar
        q1 = np.percentile(self.datos, 25)
        q2 = np.percentile(self.datos, 50)
        q3 = np.percentile(self.datos, 75)
        return [q1, q2, q3]

    def resumen_numerico(self):
        res_num = {
        'Media': self.calculo_de_media(self.datos),
        'Mediana': self.calculo_de_mediana(self.datos),
        'Desvio': self.calculo_de_desvio_estandar(self.datos),
        'Cuartiles': self.calculo_de_cuartiles(self.datos),
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

###
##################
###

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

###
##################
###

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

###
##################
###



# Clase base
class Regresion:
    def __init__(self, datos, variable_objetivo, porcentaje_entrenamiento=0.8):
        self.datos = datos.copy()
        self.variable_objetivo = variable_objetivo
        self.porcentaje_entrenamiento = porcentaje_entrenamiento
        self.resultados = None

        self._preparar_datos()

    def _preparar_datos(self):
        X = self.datos.drop(columns=[self.variable_objetivo])
        y = self.datos[self.variable_objetivo]

        n = int(len(self.datos) * self.porcentaje_entrenamiento)
        self._X_train = sm.add_constant(X.iloc[:n], has_constant='add')
        self._X_test = sm.add_constant(X.iloc[n:], has_constant='add')
        self._y_train = y.iloc[:n]
        self._y_test = y.iloc[n:]

        self._columnas_entrenamiento = self._X_train.columns

    def entrenar_modelo(self):
        raise NotImplementedError

    def predecir(self, nuevos_datos):
        raise NotImplementedError

    def evaluar_modelo(self):
        raise NotImplementedError


class RegresionLineal(Regresion):
    def entrenar_modelo(self):
        self.modelo = sm.OLS(self._y_train, self._X_train)
        self.resultados = self.modelo.fit()

    def predecir(self, nuevos_datos):
        if 'const' not in nuevos_datos.columns:
            nuevos_datos = sm.add_constant(nuevos_datos, has_constant='add')
        nuevos_datos = nuevos_datos.reindex(columns=self._columnas_entrenamiento, fill_value=0)
        return self.resultados.predict(nuevos_datos)

    def coef_correlacion(self):
        correlaciones = {}
        for col in self._X_train.columns:
            if col != 'const':
                correlaciones[col] = np.corrcoef(self._X_train[col], self._y_train)[0, 1]
        return correlaciones

    def graficar_dispersión(self):
        for col in self._X_train.columns:
            if col != 'const':
                plt.figure()
                sns.scatterplot(x=self._X_train[col], y=self._y_train)
                pred = self.resultados.predict(self._X_train)
                plt.plot(self._X_train[col], pred, color='red')
                plt.title(f'{col} vs {self.variable_objetivo}')
                plt.xlabel(col)
                plt.ylabel(self.variable_objetivo)
                plt.show()

    def analisis_residuos(self):
        residuos = self._y_test - self.predecir(self._X_test)
        predichos = self.predecir(self._X_test)

        plt.figure()
        sns.residplot(x=predichos, y=residuos, lowess=True, line_kws={'color': 'red'})
        plt.xlabel('Valores Predichos')
        plt.ylabel('Residuos')
        plt.title('Residuos vs Predichos')
        plt.show()

        plt.figure()
        stats.probplot(residuos, dist="norm", plot=plt)
        plt.title("QQ Plot")
        plt.show()

    def obtener_estadisticas(self):
        tabla = self.resultados.summary2().tables[1]
        return tabla[['Coef.', 'Std.Err.', 't', 'P>|t|']]

    def intervalos_confianza_prediccion(self):
        pred = self.resultados.get_prediction(self._X_test)
        frame = pred.summary_frame(alpha=0.05)
        return frame[['mean_ci_lower', 'mean_ci_upper', 'obs_ci_lower', 'obs_ci_upper']]

    def evaluar_modelo(self):
        y_pred = self.predecir(self._X_test)
        residuales = self._y_test - y_pred
        n = len(self._X_test)
        p = len(self._X_test.columns) - 1
        r2 = self.resultados.rsquared
        r2_ajustado = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        rmse = np.sqrt(np.mean(residuales ** 2))
        return {'R2': r2, 'R2_ajustado': r2_ajustado, 'RMSE': rmse}


class RegresionLogistica(Regresion):
    def entrenar_modelo(self):
        self.modelo = sm.Logit(self._y_train, self._X_train)
        self.resultados = self.modelo.fit()

    def predecir(self, nuevos_datos, umbral=0.5):
        if 'const' not in nuevos_datos.columns:
            nuevos_datos = sm.add_constant(nuevos_datos, has_constant='add')
        nuevos_datos = nuevos_datos.reindex(columns=self._columnas_entrenamiento, fill_value=0)
        proba = self.resultados.predict(nuevos_datos)
        return (proba >= umbral).astype(int)

    def obtener_estadisticas(self):
        tabla = self.resultados.summary2().tables[1]
        return tabla[['Coef.', 'Std.Err.', 'z', 'P>|z|']]

    def evaluar_modelo(self, umbral=0.5):
        y_pred = self.predecir(self._X_test, umbral)
        y_true = self._y_test
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        sensibilidad = tp / (tp + fn) if (tp + fn) else 0
        especificidad = tn / (tn + fp) if (tn + fp) else 0
        error_total = (fp + fn) / len(y_true)

        matriz = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=['Real 0', 'Real 1'],
            columns=['Predicho 0', 'Predicho 1']
        )

        return {
            'matriz_confusion': matriz,
            'error_total': error_total,
            'sensibilidad': sensibilidad,
            'especificidad': especificidad
        }

    def curva_roc(self):
        from sklearn.metrics import roc_curve, roc_auc_score
        proba = self.resultados.predict(self._X_test)
        fpr, tpr, _ = roc_curve(self._y_test, proba)
        auc = roc_auc_score(self._y_test, proba)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('Tasa de falsos positivos')
        plt.ylabel('Tasa de verdaderos positivos')
        plt.title('Curva ROC')
        plt.legend()
        plt.show()

        return auc

class AnalizadorCategoricos:
    def __init__(self, datos, categorias=None, esperados=None):
        """
        Inicializa el analizador con datos categóricos.

        Args:
            datos: Lista/array de observaciones categóricas
            categorias: Lista de categorías únicas (si None, se infieren)
            esperados: Valores esperados para bondad de ajuste (opcional)
        """
        self.datos = np.array(datos)
        self.categorias = categorias if categorias is not None else np.unique(self.datos)
        self.conteos = Counter(self.datos)
        self.n = len(self.datos)
        self.esperados = esperados

    def resumen_conteos(self):
        """Devuelve un DataFrame con conteos observados y proporciones."""
        df = pd.DataFrame({
            'Categoria': self.categorias,
            'Conteo': [self.conteos.get(c, 0) for c in self.categorias],
            'Proporcion': [self.conteos.get(c, 0)/self.n for c in self.categorias]
        })
        return df

    def probar_proporcion(self, categoria, valor_esperado=None, alpha=0.05, metodo='normal'):
        """
        Prueba si la proporción de una categoría difiere de un valor esperado.

        Args:
            categoria: Categoría a analizar
            valor_esperado: Valor esperado bajo H0 (si None, usa 1/k)
            alpha: Nivel de significancia
            metodo: 'normal' para aproximación normal, 'bootstrap' para bootstrap

        Returns:
            Diccionario con resultados de la prueba
        """
        if valor_esperado is None:
            valor_esperado = 1/len(self.categorias)

        p_obs = self.conteos.get(categoria, 0)/self.n
        resultados = {
            'categoria': categoria,
            'p_observada': p_obs,
            'p_esperada': valor_esperado,
            'alpha': alpha
        }

        if metodo == 'normal':
            # Aproximación normal
            se = np.sqrt(valor_esperado*(1-valor_esperado)/self.n)
            z = (p_obs - valor_esperado)/se
            p_valor = 2*(1 - norm.cdf(abs(z)))

            ic_inf = p_obs - norm.ppf(1-alpha/2)*se
            ic_sup = p_obs + norm.ppf(1-alpha/2)*se

            resultados.update({
                'metodo': 'aproximacion_normal',
                'estadistico': z,
                'p_valor': p_valor,
                'ic_inferior': ic_inf,
                'ic_superior': ic_sup
            })

        elif metodo == 'bootstrap':
            # Método bootstrap
            B = 5000
            p_boot = []
            for _ in range(B):
                muestra = np.random.choice(self.datos, size=self.n, replace=True)
                p_boot.append(np.sum(muestra == categoria)/self.n)

            se_boot = np.std(p_boot)
            ic_inf = np.percentile(p_boot, 100*alpha/2)
            ic_sup = np.percentile(p_boot, 100*(1-alpha/2))

            resultados.update({
                'metodo': 'bootstrap',
                'se_bootstrap': se_boot,
                'ic_inferior': ic_inf,
                'ic_superior': ic_sup
            })

        return resultados

    def prueba_bondad_ajuste(self, esperados=None, alpha=0.05):
        """
        Realiza prueba χ² de bondad de ajuste.

        Args:
            esperados: Valores esperados (si None, asume distribución uniforme)
            alpha: Nivel de significancia

        Returns:
            Diccionario con resultados de la prueba
        """
        if esperados is None:
            esperados = [self.n/len(self.categorias)] * len(self.categorias)
        elif isinstance(esperados, (list, np.ndarray)):
            esperados = np.array(esperados) * self.n

        observados = np.array([self.conteos.get(c, 0) for c in self.categorias])

        # Calcular estadístico χ²
        chi2_obs = np.sum((observados - esperados)**2 / esperados)
        df = len(self.categorias) - 1
        p_valor = 1 - chi2.cdf(chi2_obs, df)
        chi2_critico = chi2.ppf(1 - alpha, df)

        return {
            'estadistico': chi2_obs,
            'grados_libertad': df,
            'p_valor': p_valor,
            'valor_critico': chi2_critico,
            'rechazar_H0': p_valor < alpha,
            'observados': observados,
            'esperados': esperados
        }

    def graficar_conteos(self):
        """Grafica conteos observados vs esperados."""
        res = self.prueba_bondad_ajuste()
        fig, ax = plt.subplots(figsize=(10, 5))

        x = np.arange(len(self.categorias))
        ancho = 0.35

        ax.bar(x - ancho/2, res['observados'], ancho, label='Observados')
        ax.bar(x + ancho/2, res['esperados'], ancho, label='Esperados')

        ax.set_xticks(x)
        ax.set_xticklabels(self.categorias)
        ax.set_ylabel('Frecuencia')
        ax.set_title('Conteos Observados vs Esperados')
        ax.legend()

        plt.show()
        return fig
