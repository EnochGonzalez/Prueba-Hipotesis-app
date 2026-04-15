import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="Prueba de Hipótesis", layout="wide")
st.title("📊 App de Prueba de Hipótesis")

# ── SIDEBAR ──────────────────────────────────────────────
st.sidebar.header("Configuración")

# ── MÓDULO 1: CARGA DE DATOS ─────────────────────────────
st.header("1. Datos")
fuente = st.radio("Fuente de datos", ["Generar datos sintéticos", "Subir CSV"])

if fuente == "Generar datos sintéticos":
    n = st.slider("Número de observaciones", 30, 500, 100)
    media_real = st.number_input("Media real de la población", value=0.0)
    sigma = st.number_input("Desviación estándar", value=1.0, min_value=0.01)
    semilla = st.number_input("Semilla aleatoria", value=42, step=1)
    np.random.seed(int(semilla))
    datos = np.random.normal(loc=media_real, scale=sigma, size=n)
    df = pd.DataFrame({"valor": datos})
else:
    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if archivo:
        df = pd.read_csv(archivo)
        col = st.selectbox("Selecciona la columna numérica", df.select_dtypes(include=np.number).columns)
        df = pd.DataFrame({"valor": df[col].dropna()})
    else:
        st.stop()

st.subheader("Vista previa de los datos")
st.dataframe(df.describe().T, use_container_width=True)

# ── MÓDULO 2: VISUALIZACIONES ─────────────────────────────
st.header("2. Distribución de los datos")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["valor"], kde=True, ax=ax, color="#4C72B0")
    ax.set_title("Histograma con KDE")
    ax.set_xlabel("Valor")
    st.pyplot(fig)

with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.boxplot(y=df["valor"], ax=ax2, color="#55A868")
    ax2.set_title("Boxplot")
    st.pyplot(fig2)

# Preguntas de normalidad
st.subheader("Análisis visual")
st.write(f"**¿Distribución parece normal?** Media={df['valor'].mean():.3f}, Sesgo={df['valor'].skew():.3f}")
if abs(df["valor"].skew()) < 0.5:
    st.success("✅ La distribución parece aproximadamente normal (sesgo bajo).")
else:
    st.warning(f"⚠️ La distribución muestra sesgo de {df['valor'].skew():.3f}.")

q1, q3 = df["valor"].quantile([0.25, 0.75])
iqr = q3 - q1
outliers = df[(df["valor"] < q1 - 1.5*iqr) | (df["valor"] > q3 + 1.5*iqr)]
st.write(f"**Outliers detectados:** {len(outliers)} de {len(df)} observaciones.")

# ── MÓDULO 3: PRUEBA DE HIPÓTESIS ────────────────────────
st.header("3. Prueba de Hipótesis (prueba Z)")

col_a, col_b = st.columns(2)
with col_a:
    mu0 = st.number_input("Hipótesis nula H₀: μ =", value=0.0)
    sigma_pob = st.number_input("Desviación estándar poblacional (σ)", value=1.0, min_value=0.01)
    alpha = st.selectbox("Nivel de significancia α", [0.01, 0.05, 0.10], index=1)

with col_b:
    tipo_prueba = st.selectbox("Tipo de prueba", ["Bilateral (≠)", "Cola izquierda (<)", "Cola derecha (>)"])

if st.button("▶ Ejecutar prueba Z"):
    n_muestra = len(df)
    x_bar = df["valor"].mean()
    Z = (x_bar - mu0) / (sigma_pob / np.sqrt(n_muestra))

    # p-value según tipo
    if tipo_prueba == "Bilateral (≠)":
        p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
        z_critico = stats.norm.ppf(1 - alpha/2)
        rechazar = abs(Z) > z_critico
    elif tipo_prueba == "Cola izquierda (<)":
        p_value = stats.norm.cdf(Z)
        z_critico = stats.norm.ppf(alpha)
        rechazar = Z < z_critico
    else:
        p_value = 1 - stats.norm.cdf(Z)
        z_critico = stats.norm.ppf(1 - alpha)
        rechazar = Z > z_critico

    # Resultados
    st.subheader("Resultados")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Media muestral (x̄)", f"{x_bar:.4f}")
    r2.metric("Estadístico Z", f"{Z:.4f}")
    r3.metric("p-value", f"{p_value:.4f}")
    r4.metric("n", n_muestra)

    if rechazar:
        st.error(f"🔴 Se **rechaza H₀** (Z={Z:.3f}, p={p_value:.4f} < α={alpha})")
    else:
        st.success(f"🟢 **No se rechaza H₀** (Z={Z:.3f}, p={p_value:.4f} ≥ α={alpha})")

    # Gráfica de la curva con zona de rechazo
    st.subheader("Curva normal con zona de rechazo")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    x_vals = np.linspace(-4, 4, 400)
    y_vals = stats.norm.pdf(x_vals)
    ax3.plot(x_vals, y_vals, "k-", lw=2)

    # Zonas de rechazo
    if tipo_prueba == "Bilateral (≠)":
        ax3.fill_between(x_vals, y_vals, where=(x_vals <= -z_critico), color="red", alpha=0.3, label="Zona de rechazo")
        ax3.fill_between(x_vals, y_vals, where=(x_vals >= z_critico), color="red", alpha=0.3)
        ax3.axvline(-z_critico, color="red", linestyle="--", label=f"±Z_c = ±{z_critico:.3f}")
        ax3.axvline(z_critico, color="red", linestyle="--")
    elif tipo_prueba == "Cola izquierda (<)":
        ax3.fill_between(x_vals, y_vals, where=(x_vals <= z_critico), color="red", alpha=0.3, label="Zona de rechazo")
        ax3.axvline(z_critico, color="red", linestyle="--", label=f"Z_c = {z_critico:.3f}")
    else:
        ax3.fill_between(x_vals, y_vals, where=(x_vals >= z_critico), color="red", alpha=0.3, label="Zona de rechazo")
        ax3.axvline(z_critico, color="red", linestyle="--", label=f"Z_c = {z_critico:.3f}")

    ax3.axvline(Z, color="blue", linestyle="-", lw=2, label=f"Z calculado = {Z:.3f}")
    ax3.fill_between(x_vals, y_vals, where=((x_vals > -z_critico) & (x_vals < z_critico)) if tipo_prueba == "Bilateral (≠)" else (x_vals > z_critico if tipo_prueba == "Cola derecha (>)" else x_vals < z_critico), color="green", alpha=0.1, label="Zona de no rechazo")
    ax3.set_title("Distribución normal estándar")
    ax3.legend()
    ax3.set_xlabel("Z")
    ax3.set_ylabel("Densidad")
    st.pyplot(fig3)

    # ── MÓDULO 4: IA con Gemini ───────────────────────────
    st.header("4. Interpretación con IA (Gemini)")

    prompt_ia = f"""Se realizó una prueba Z con los siguientes parámetros:
- Media muestral: {x_bar:.4f}
- Media hipotética (H₀): {mu0}
- Tamaño de muestra: {n_muestra}
- Desviación estándar poblacional: {sigma_pob}
- Nivel de significancia: {alpha}
- Tipo de prueba: {tipo_prueba}
- Estadístico Z calculado: {Z:.4f}
- p-value: {p_value:.4f}
- Sesgo de los datos: {df['valor'].skew():.4f}

¿Se rechaza H₀? Explica la decisión estadística en español, indica si los supuestos de la prueba Z son razonables, y menciona qué implica este resultado en la práctica."""

    with st.spinner("Consultando a Gemini..."):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt_ia)
            respuesta_ia = response.text

            st.subheader("Respuesta de Gemini")
            st.write(respuesta_ia)

            st.subheader("Comparación: tu decisión vs. IA")
            tu_decision = "Rechazar H₀" if rechazar else "No rechazar H₀"
            ia_rechaza = "rechaz" in respuesta_ia.lower()
            ia_decision = "Rechazar H₀" if ia_rechaza else "No rechazar H₀"
            c1, c2 = st.columns(2)
            c1.info(f"**Tu decisión (automática):** {tu_decision}")
            c2.info(f"**Decisión de la IA:** {ia_decision}")

            # Guardar prompt para el reporte
            st.subheader("📋 Prompt enviado a Gemini (cópialo para tu reporte)")
            st.code(prompt_ia)

        except Exception as e:
            st.error(f"Error al conectar con Gemini: {e}")