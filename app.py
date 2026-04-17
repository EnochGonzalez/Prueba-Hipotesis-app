import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from dotenv import load_dotenv
import os

# --- CONFIGURACIÓN ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

if 'ejecutado' not in st.session_state:
    st.session_state['ejecutado'] = False

st.set_page_config(page_title="Prueba de Hipótesis", layout="wide", page_icon="📊")

# --- HEADER ---
st.markdown("""
    <h1 style='text-align: center; color: #4C72B0;'>📊 App de Prueba de Hipótesis</h1>
    <p style='text-align: center; color: gray;'>Visualización · Estadística · Inteligencia Artificial</p>
    <hr>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## ⚙️ Navegación")
    st.markdown("Sigue los módulos de arriba hacia abajo:")
    st.markdown("1. 📂 Carga de datos")
    st.markdown("2. 📈 Visualización")
    st.markdown("3. 🧪 Prueba Z")
    st.markdown("4. 🤖 IA con Gemini")
    st.markdown("---")
    st.markdown("### 📖 Glosario rápido")
    st.info("**Media (μ):** Promedio de todos los datos.")
    st.info("**Desv. estándar (σ):** Qué tan dispersos están los datos respecto a la media.")
    st.info("**p-value:** Probabilidad de obtener tu resultado si H₀ fuera verdadera. Muy pequeño = evidencia en contra de H₀.")
    st.info("**α:** Umbral de decisión. Si p < α, rechazas H₀.")
    st.markdown("---")
    st.caption("Desarrollado por Santos González · 2026")

# =============================================
# MÓDULO 1: CARGA DE DATOS
# =============================================
st.markdown("## 📂 1. Carga de Datos")

with st.expander("💡 ¿Qué hago aquí?", expanded=False):
    st.markdown("""
    En este módulo defines **de dónde vienen tus datos**:
    - **Datos sintéticos:** La app los genera automáticamente con una distribución normal. 
      Útil para probar la app sin tener datos reales.
    - **CSV:** Sube tu propio archivo con datos reales (por ejemplo, resultados de una encuesta).
    
    El tamaño de muestra debe ser **n ≥ 30** para que la prueba Z sea válida.
    """)

fuente = st.radio("Selecciona el origen de los datos:",
                  ["Generar datos sintéticos", "Subir CSV"],
                  horizontal=True)

df = None

if fuente == "Generar datos sintéticos":
    with st.expander("⚙️ Parámetros de generación", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            n = st.slider("Observaciones (n)", 30, 500, 100)
        with col2:
            media_real = st.number_input("Media real (μ)", value=50.0)
        with col3:
            sigma = st.number_input("Desv. estándar (σ)", value=10.0, min_value=0.01)
        with col4:
            semilla = st.number_input("Semilla aleatoria", value=42, step=1)

    np.random.seed(int(semilla))
    datos = np.random.normal(loc=media_real, scale=sigma, size=n)
    df = pd.DataFrame({"valor": datos})
    st.success(f"✅ Se generaron **{n} datos** con media={media_real} y σ={sigma}")

else:
    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if archivo:
        df_raw = pd.read_csv(archivo)
        col = st.selectbox("Selecciona la columna numérica:",
                           df_raw.select_dtypes(include=np.number).columns)
        df = pd.DataFrame({"valor": df_raw[col].dropna()})
        st.success(f"✅ CSV cargado — {len(df)} observaciones en la columna '{col}'")
    else:
        st.info("👆 Sube un archivo CSV para continuar.")
        st.stop()

with st.expander("📋 Ver estadísticos descriptivos"):
    st.markdown("""
    **¿Cómo leer esta tabla?**
    - **count:** número de datos
    - **mean:** promedio
    - **std:** desviación estándar (dispersión)
    - **min/max:** valores extremos
    - **25%, 50%, 75%:** cuartiles (cómo se distribuyen los datos)
    """)
    desc = df.describe().T
    st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)

st.markdown("---")

# =============================================
# MÓDULO 2: VISUALIZACIÓN
# =============================================
st.markdown("## 📈 2. Visualización de la Distribución")

with st.expander("💡 ¿Cómo leer estas gráficas?", expanded=False):
    st.markdown("""
    ### 📊 Histograma con KDE
    - Las **barras** muestran cuántos datos caen en cada rango de valores.
    - La **curva KDE** (línea suave) estima la forma de la distribución.
    - Si la curva tiene forma de **campana simétrica**, la distribución es aproximadamente normal.
    - Si la curva está cargada hacia un lado, hay **sesgo**.

    ### 📦 Boxplot (Caja y bigotes)
    - La **línea roja del centro** es la mediana (el valor medio).
    - La **caja** contiene el 50% central de los datos (del cuartil 25% al 75%).
    - Los **bigotes** se extienden hasta 1.5 veces el rango intercuartílico.
    - Los **puntos fuera de los bigotes** son **outliers** (valores atípicos).
    """)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["valor"], kde=True, ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_title("Histograma con KDE", fontsize=13, fontweight="bold")
    ax.set_xlabel("Valor")
    ax.set_ylabel("Frecuencia")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    plt.close()

with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.boxplot(y=df["valor"], ax=ax2, color="#55A868",
                medianprops=dict(color="red", linewidth=2))
    ax2.set_title("Boxplot — Detección de Outliers", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Valor")
    ax2.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig2)
    plt.close()

# Análisis automático
st.markdown("### 🔍 Análisis automático de la distribución")

with st.expander("💡 ¿Qué significan estos indicadores?", expanded=False):
    st.markdown("""
    - **Sesgo (Skewness):** Mide la asimetría de la distribución.
        - Sesgo ≈ 0 → distribución simétrica (ideal para prueba Z)
        - Sesgo > 0 → cola larga hacia la derecha
        - Sesgo < 0 → cola larga hacia la izquierda
    - **Outliers:** Datos que se alejan mucho del resto. Pueden afectar los resultados estadísticos.
    """)

media_m = df["valor"].mean()
sesgo = df["valor"].skew()
q1, q3 = df["valor"].quantile([0.25, 0.75])
iqr = q3 - q1
outliers = df[(df["valor"] < q1 - 1.5 * iqr) | (df["valor"] > q3 + 1.5 * iqr)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Media muestral (x̄)", f"{media_m:.3f}")
c2.metric("Sesgo", f"{sesgo:.3f}")
c3.metric("Outliers detectados", f"{len(outliers)}")
c4.metric("Total de datos (n)", f"{len(df)}")

if abs(sesgo) < 0.5:
    st.success("✅ La distribución parece **aproximadamente normal** (sesgo bajo). La prueba Z es apropiada.")
elif abs(sesgo) < 1.0:
    st.warning(f"⚠️ La distribución tiene **sesgo moderado** ({sesgo:.3f}). Úsala con precaución.")
else:
    st.error(f"❌ La distribución tiene **sesgo alto** ({sesgo:.3f}). Considera si la prueba Z es adecuada.")

if len(outliers) > 0:
    st.warning(f"⚠️ Se detectaron **{len(outliers)} outliers** que podrían afectar los resultados.")
else:
    st.success("✅ No se detectaron outliers significativos.")

st.markdown("#### ✍️ Responde tú:")
col_resp1, col_resp2 = st.columns(2)
with col_resp1:
    st.radio("¿La distribución parece normal?",
             ["Sin responder", "Sí, parece normal", "No, no parece normal"],
             key="preg_normal")
with col_resp2:
    st.text_area("Observaciones sobre sesgo y outliers:",
                 placeholder="Ejemplo: La distribución parece normal con sesgo bajo. No hay outliers visibles...",
                 key="preg_outliers")

st.markdown("---")

# =============================================
# MÓDULO 3: PRUEBA Z
# =============================================
st.markdown("## 🧪 3. Prueba de Hipótesis (Prueba Z)")

with st.expander("💡 ¿Qué es la prueba Z y cómo funciona?", expanded=False):
    st.markdown("""
    La **prueba Z** sirve para verificar si la media de tu muestra es significativamente 
    diferente de un valor hipotético (H₀).

    **Fórmula:**
    > Z = (x̄ − μ₀) / (σ / √n)

    **¿Cuándo usarla?**
    - Varianza poblacional **conocida**
    - Tamaño de muestra **n ≥ 30**

    **Tipos de prueba:**
    - **Bilateral (≠):** ¿La media es diferente (mayor O menor)?
    - **Cola izquierda (<):** ¿La media es menor que H₀?
    - **Cola derecha (>):** ¿La media es mayor que H₀?

    **¿Qué significa rechazar H₀?**
    Si rechazas H₀, hay evidencia estadística suficiente para decir que 
    la media real **no es** el valor que planteaste. Si no rechazas, 
    no hay suficiente evidencia para descartarla.
    """)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("**Hipótesis nula**")
    mu0 = st.number_input("H₀: μ =", value=50.0)
    sigma_pob = st.number_input("Desv. estándar poblacional (σ)", value=10.0, min_value=0.01)
with col_b:
    st.markdown("**Hipótesis alternativa**")
    tipo_prueba = st.selectbox("H₁:",
                               ["Bilateral (μ ≠ μ₀)",
                                "Cola izquierda (μ < μ₀)",
                                "Cola derecha (μ > μ₀)"])
with col_c:
    st.markdown("**Nivel de significancia**")
    alpha = st.selectbox("α =", [0.01, 0.05, 0.10], index=1)
    st.caption(f"Nivel de confianza: **{(1 - alpha) * 100:.0f}%**")
    st.caption(f"Con α={alpha}: si p < {alpha}, se rechaza H₀.")

if st.button("▶ Ejecutar Prueba Z", type="primary", use_container_width=True):
    n_muestra = len(df)
    x_bar = df["valor"].mean()
    Z = (x_bar - mu0) / (sigma_pob / np.sqrt(n_muestra))

    if "Bilateral" in tipo_prueba:
        p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
        z_critico = stats.norm.ppf(1 - alpha / 2)
        rechazar = abs(Z) > z_critico
    elif "izquierda" in tipo_prueba:
        p_value = stats.norm.cdf(Z)
        z_critico = stats.norm.ppf(alpha)
        rechazar = Z < z_critico
    else:
        p_value = 1 - stats.norm.cdf(Z)
        z_critico = stats.norm.ppf(1 - alpha)
        rechazar = Z > z_critico

    st.session_state.update({
        'z_stat': Z, 'p_val': p_value, 'rechazo': rechazar,
        'n_m': n_muestra, 'x_bar': x_bar, 'h0_val': mu0,
        'sigma_pob': sigma_pob, 'tipo_prueba': tipo_prueba,
        'z_critico': z_critico, 'alpha': alpha, 'ejecutado': True
    })

if st.session_state['ejecutado']:
    Z = st.session_state['z_stat']
    p_value = st.session_state['p_val']
    rechazar = st.session_state['rechazo']
    z_critico = st.session_state['z_critico']
    tipo_prueba = st.session_state['tipo_prueba']
    x_bar = st.session_state['x_bar']
    alpha = st.session_state['alpha']

    st.markdown("### 📊 Resultados de la prueba")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Media muestral (x̄)", f"{x_bar:.4f}")
    r2.metric("Estadístico Z", f"{Z:.4f}")
    r3.metric("p-value", f"{p_value:.4f}")
    r4.metric("Z crítico", f"±{z_critico:.4f}" if "Bilateral" in tipo_prueba else f"{z_critico:.4f}")

    if rechazar:
        st.error(f"🔴 **Se rechaza H₀** — Z={Z:.3f} cae en la zona de rechazo (p={p_value:.4f} < α={alpha})")
    else:
        st.success(f"🟢 **No se rechaza H₀** — Z={Z:.3f} no cae en la zona de rechazo (p={p_value:.4f} ≥ α={alpha})")

    with st.expander("💡 ¿Cómo interpretar estos resultados?", expanded=False):
        st.markdown(f"""
        - **Z calculado = {Z:.4f}:** Indica cuántas desviaciones estándar está tu media muestral 
          respecto al valor de H₀. Mientras más alejado de 0, más evidencia en contra de H₀.
        - **p-value = {p_value:.4f}:** Es la probabilidad de obtener un Z tan extremo si H₀ fuera verdadera.
          {'Como p < α, hay evidencia suficiente para rechazar H₀.' if rechazar else 'Como p ≥ α, no hay suficiente evidencia para rechazar H₀.'}
        - **Z crítico = ±{z_critico:.4f}:** Es el umbral. Si tu Z calculado lo supera, cae en zona de rechazo.
        
        > **Nota:** Un Z muy grande (ej. |Z| > 10) simplemente significa que los datos son 
        muy diferentes de H₀. No es un error, ¡es la prueba funcionando correctamente!
        """)

    # Gráfica con zona de rechazo
    st.markdown("### 📉 Curva Normal con Zona de Rechazo")

    with st.expander("💡 ¿Cómo leer esta gráfica?", expanded=False):
        st.markdown("""
        - **Zona roja:** Zona de rechazo. Si tu Z calculado cae aquí, rechazas H₀.
        - **Zona verde:** Zona de no rechazo. Si tu Z calculado cae aquí, no rechazas H₀.
        - **Línea azul sólida:** Tu estadístico Z calculado.
        - **Línea roja punteada:** El valor crítico Z_c que define la frontera de decisión.
        - El área bajo la curva fuera de las líneas rojas equivale exactamente a α.
        """)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    x_vals = np.linspace(-4, 4, 400)
    y_vals = stats.norm.pdf(x_vals)
    ax3.plot(x_vals, y_vals, "k-", lw=2)

    if "Bilateral" in tipo_prueba:
        ax3.fill_between(x_vals, y_vals, where=(abs(x_vals) >= z_critico),
                         color="red", alpha=0.35, label="Zona de rechazo")
        ax3.fill_between(x_vals, y_vals, where=(abs(x_vals) < z_critico),
                         color="green", alpha=0.1, label="Zona de no rechazo")
        ax3.axvline(-z_critico, color="red", linestyle="--", lw=1.5)
        ax3.axvline(z_critico, color="red", linestyle="--", lw=1.5,
                    label=f"Z crítico = ±{z_critico:.3f}")
    elif "izquierda" in tipo_prueba:
        ax3.fill_between(x_vals, y_vals, where=(x_vals <= z_critico),
                         color="red", alpha=0.35, label="Zona de rechazo")
        ax3.fill_between(x_vals, y_vals, where=(x_vals > z_critico),
                         color="green", alpha=0.1, label="Zona de no rechazo")
        ax3.axvline(z_critico, color="red", linestyle="--", lw=1.5,
                    label=f"Z crítico = {z_critico:.3f}")
    else:
        ax3.fill_between(x_vals, y_vals, where=(x_vals >= z_critico),
                         color="red", alpha=0.35, label="Zona de rechazo")
        ax3.fill_between(x_vals, y_vals, where=(x_vals < z_critico),
                         color="green", alpha=0.1, label="Zona de no rechazo")
        ax3.axvline(z_critico, color="red", linestyle="--", lw=1.5,
                    label=f"Z crítico = {z_critico:.3f}")

    # Clamp Z para que aparezca en la gráfica aunque sea muy grande
    z_plot = max(min(Z, 3.9), -3.9)
    ax3.axvline(z_plot, color="blue", linestyle="-", lw=2.5,
                label=f"Z calculado = {Z:.3f}" + (" (fuera de rango visible)" if abs(Z) > 4 else ""))

    ax3.set_title("Distribución Normal Estándar — Zona de Rechazo", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Z")
    ax3.set_ylabel("Densidad")
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.legend()
    st.pyplot(fig3)
    plt.close()

    if abs(Z) > 4:
        st.info(f"ℹ️ Tu Z calculado ({Z:.3f}) está fuera del rango visible de la gráfica (-4 a 4), "
                f"pero la decisión es correcta: cae completamente en la zona de rechazo.")

st.markdown("---")

# =============================================
# MÓDULO 4: IA CON GEMINI
# =============================================
st.markdown("## 🤖 4. Interpretación con IA (Gemini)")

with st.expander("💡 ¿Para qué sirve este módulo?", expanded=False):
    st.markdown("""
    Este módulo envía un **resumen estadístico** (no los datos crudos) a la IA de Google Gemini,
    que actúa como un experto en estadística y te explica:
    1. Si la decisión de rechazar o no H₀ es correcta.
    2. Qué significa el resultado en términos prácticos.
    3. Si los supuestos de la prueba Z son razonables.

    Luego puedes **comparar** la decisión automática de la app con la interpretación de la IA.
    Esto es útil para verificar y aprender.
    """)

if not st.session_state.get('ejecutado'):
    st.info("⬆️ Primero ejecuta la Prueba Z en el módulo 3 para habilitar la consulta a la IA.")
else:
    Z = st.session_state['z_stat']
    p = st.session_state['p_val']
    n = st.session_state['n_m']
    h0 = st.session_state['h0_val']
    sigma = st.session_state['sigma_pob']
    alpha = st.session_state['alpha']
    tipo = st.session_state['tipo_prueba']
    decision = "Rechazar H₀" if st.session_state['rechazo'] else "No Rechazar H₀"

    prompt_ia = f"""Actúa como un experto en estadística explicando a un estudiante de ingeniería de software.
Se realizó una prueba Z con los siguientes parámetros:
- Tamaño de muestra (n): {n}
- Media muestral (x̄): {st.session_state['x_bar']:.4f}
- Media hipotética (H₀): {h0}
- Desviación estándar poblacional (σ): {sigma}
- Nivel de significancia (α): {alpha}
- Tipo de prueba: {tipo}
- Estadístico Z calculado: {Z:.4f}
- Valor p: {p:.4f}
- Decisión automática: {decision}

Explica en español y en lenguaje sencillo:
1. Si la decisión estadística es correcta y por qué.
2. Qué significa este resultado en la práctica.
3. Si los supuestos de la prueba Z son razonables con estos datos."""

    with st.expander("📋 Ver prompt enviado a Gemini"):
        st.code(prompt_ia)

    if st.button("🤖 Consultar a Gemini", type="primary", use_container_width=True):
        with st.spinner("Consultando a Gemini..."):
            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(prompt_ia)
                if response.text:
                    st.markdown("### 🤖 Interpretación de la IA:")
                    st.write(response.text)
                    st.success("✅ Interpretación generada con éxito.")

                    decision_auto = "Rechazar H₀" if st.session_state['rechazo'] else "No rechazar H₀"
                    ia_rechaza = "rechaz" in response.text.lower()
                    decision_ia = "Rechazar H₀" if ia_rechaza else "No rechazar H₀"

                    st.markdown("### ⚖️ Comparación de decisiones")
                    ca, cb = st.columns(2)
                    ca.info(f"**Decisión automática de la app:** {decision_auto}")
                    cb.info(f"**Decisión interpretada de la IA:** {decision_ia}")

                    if decision_auto == decision_ia:
                        st.success("✅ Ambas decisiones coinciden. La prueba es consistente.")
                    else:
                        st.warning("⚠️ Las decisiones no coinciden. Analiza el porqué en tu reporte — "
                                   "esto puede ser material valioso para la sección de reflexión.")

            except Exception as e:
                st.error(f"Error: {e}")
                st.warning("Verifica tu API Key en: https://aistudio.google.com/")