import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA Y ESTILO
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Predictor de Resistencia — Hormigón Proyectado",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #F8FAFC; }
    h1 { color: #1E293B; font-weight: 700; letter-spacing: -0.5px; }
    h2, h3 { color: #334155; font-weight: 600; }

    /* Barra lateral */
    section[data-testid="stSidebar"] { background-color: #1E293B; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p { color: #F1F5F9 !important; }

    /* Campos de entrada de la barra lateral: fondo claro + texto oscuro */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] div[data-baseweb="input"],
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }
    section[data-testid="stSidebar"] input { color: #1E293B !important; }
    section[data-testid="stSidebar"] div[data-baseweb="select"] * { color: #1E293B !important; }
    section[data-testid="stSidebar"] button { color: #1E293B !important; }

    /* Tarjetas de métricas */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] { color: #64748B; font-weight: 500; }
    div[data-testid="stMetricValue"] { color: #1E293B; font-weight: 700; }

    hr { border-color: #E2E8F0; }
    .header-band {
        background: linear-gradient(90deg, #2563EB 0%, #1E40AF 100%);
        height: 5px; border-radius: 3px; margin-bottom: 18px;
    }
    .fila-modelo {
        display: flex; align-items: center; height: 100%;
        font-weight: 700; color: #1E293B; font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 1. CARGA DE MODELOS (modelo único con el tiempo como variable)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def cargar_modelos():
    p_rf, p_xgb = None, None
    try:
        p_rf = joblib.load('modelo_hormigon_rf.pkl')
    except Exception as e:
        st.error(f"No se pudo cargar 'modelo_hormigon_rf.pkl': {e}")
    try:
        p_xgb = joblib.load('modelo_hormigon_xgb.pkl')
    except Exception as e:
        st.error(f"No se pudo cargar 'modelo_hormigon_xgb.pkl': {e}")
    return p_rf, p_xgb

paquete_rf, paquete_xgb = cargar_modelos()

paquete_ref = paquete_rf or paquete_xgb
if paquete_ref is None:
    st.stop()

# ── Comprobación de arquitectura: este .pkl debe ser el del MODELO ÚNICO ──────
if 'modelo_unico' not in paquete_ref:
    st.error(
        "El archivo .pkl cargado no contiene la clave 'modelo_unico'. "
        "Esta app espera los modelos de la **Celda 2.B** (modelo único con el "
        "tiempo como variable, solo calorimetría). Reexporta los .pkl desde esa celda."
    )
    st.stop()

# ── Leer equivalencias y combinaciones desde el .pkl (no hardcoded) ──────────
factorizacion  = paquete_ref.get('factorizacion', {})
combos_validos = paquete_ref.get('combos_validos', {})

def tabla_texto_a_codigo(col):
    """Devuelve {texto: codigo} para una columna factorizada."""
    lista = factorizacion.get(col, [])
    return {str(txt): i for i, txt in enumerate(lista)}

COD_TIPO = tabla_texto_a_codigo('tipo_acelerante')
COD_ACE  = tabla_texto_a_codigo('acelerante')
COD_CEM  = tabla_texto_a_codigo('cemento')
COD_MAT  = tabla_texto_a_codigo('material')

cod_a_nombre_ace = {i: str(txt) for i, txt in enumerate(factorizacion.get('acelerante', []))}
FAMILIAS = {}
for familia, codigos_prod in combos_validos.items():
    FAMILIAS[familia] = [cod_a_nombre_ace[c] for c in codigos_prod if c in cod_a_nombre_ace]

# ═══════════════════════════════════════════════════════════════════════════
# 2. CABECERA
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="header-band"></div>', unsafe_allow_html=True)
st.title("Predictor de Resistencia — Hormigón Proyectado")
st.markdown("Ajuste los parámetros de diseño y los datos de laboratorio para "
            "contrastar las curvas predictivas de los modelos de Machine Learning.")

valores = {}

# ═══════════════════════════════════════════════════════════════════════════
# 3. BARRA LATERAL — DATOS DE ENSAYO
# ═══════════════════════════════════════════════════════════════════════════
st.sidebar.header("1 · Datos de Ensayo Temprano")
st.sidebar.caption("Este modelo usa únicamente variables de **calorimetría** "
                   "(13 variables, sin temperatura).")

# Inicialización de las 7 variables de calorimetría que usa el modelo de 13 vars
for v in ['punto_max_ace_mWg', 'pendiente_ace_mWgh', 'energia_ace_Jg',
          'punto_max_ppal_mWg', 'pendiente_ppal_mWgh', 'energia_ppal_Jg',
          'energia_total']:
    valores[v] = 0.0

st.sidebar.subheader("Variables de Calorimetría")
valores['punto_max_ace_mWg']   = st.sidebar.number_input("Punto máx Ace (mW/g)", value=40.0)
valores['pendiente_ace_mWgh']  = st.sidebar.number_input("Pendiente 1 (mW/g·h)", value=0.5)
valores['energia_ace_Jg']      = st.sidebar.number_input("Energía 1 (J/g)", value=10.0)
valores['punto_max_ppal_mWg']  = st.sidebar.number_input("Punto máx Principal (mW/g)", value=4.5)
valores['pendiente_ppal_mWgh'] = st.sidebar.number_input("Pendiente 2 (mW/g·h)", value=0.4)
valores['energia_ppal_Jg']     = st.sidebar.number_input("Energía 2 (J/g)", value=15.0)
valores['energia_total']       = st.sidebar.number_input("Energía Total 1 (J/g)", value=25.0)

# ═══════════════════════════════════════════════════════════════════════════
# 4. CUERPO PRINCIPAL — DISEÑO DE LA MEZCLA
# ═══════════════════════════════════════════════════════════════════════════
st.header("Parámetros de la Mezcla")
col1, col2, col3 = st.columns(3)

with col1:
    nombres_material = list(COD_MAT.keys()) or ["Mortero", "Pasta"]
    mat_sel = st.selectbox("Material", nombres_material)
    valores['material'] = COD_MAT.get(mat_sel, 0)

    nombres_cemento = list(COD_CEM.keys()) or ["CEM I 52,5 R"]
    cem_sel = st.selectbox("Tipo de Cemento", nombres_cemento)
    valores['cemento'] = COD_CEM.get(cem_sel, 0)

with col2:
    valores['relacion_ac'] = st.number_input(
        "Relación Agua/Cemento", min_value=0.30, max_value=0.60, value=0.42, step=0.01)

    familia_sel = st.selectbox("Familia Acelerante", list(FAMILIAS.keys()))
    valores['tipo_acelerante'] = COD_TIPO.get(familia_sel, 0)

with col3:
    valores['dosificacion'] = st.number_input(
        "Dosificación Acelerante (%)", min_value=0.0, max_value=12.0, value=5.0, step=0.1)

    productos_validos = FAMILIAS.get(familia_sel, [])
    producto_sel = st.selectbox("Producto Acelerante", productos_validos,
                                help="Solo se muestran los productos compatibles "
                                     "con la familia seleccionada.")
    valores['acelerante'] = COD_ACE.get(producto_sel, 0)

st.caption(f"Acelerante: **{familia_sel} / {producto_sel}**")

# ═══════════════════════════════════════════════════════════════════════════
# 5. CONTROLES SOBRE EL GRÁFICO
# ═══════════════════════════════════════════════════════════════════════════
col_izq, col_med, col_der = st.columns([2, 2, 2])
with col_izq:
    suavizar = st.checkbox("Suavizar curvas (PCHIP)", value=True)
with col_med:
    monotona = st.checkbox("Forzar curva creciente", value=True,
                           help="La resistencia del hormigón nunca decrece "
                                "con el tiempo (regresión isotónica).")
with col_der:
    sub1, sub2 = st.columns(2)
    with sub1:
        mostrar_rf = st.checkbox("Random Forest", value=True) if paquete_rf else False
    with sub2:
        mostrar_xgb = st.checkbox("XGBoost", value=True) if paquete_xgb else False

col_a, _ = st.columns([2, 4])
with col_a:
    mostrar_crudo = st.checkbox(
        "Mostrar predicción cruda (escalones)", value=True,
        help="Superpone la salida directa del modelo (escalonada, el 'valor "
             "verdadero' que predice) sobre la curva procesada, para ver el "
             "efecto de la monotonía + PCHIP.")

# ═══════════════════════════════════════════════════════════════════════════
# 6. MOTOR DE PREDICCIÓN  ·  UN SOLO MODELO, BARRIDO EN EL TIEMPO
# ═══════════════════════════════════════════════════════════════════════════
# Eje temporal continuo: 3 min → 28 d (mismo rango que en la Celda 2 nueva).
HORAS_EJE = np.logspace(np.log10(3/60), np.log10(672), 300)

# Número de nodos PCHIP FIJO (el suavizado ya no es variable desde la interfaz).
N_ANCLAS = 25

def predecir(paquete, horas_eje=HORAS_EJE):
    """
    El modelo único se entrenó sobre [features_ok..., tiempo_horas] con el tiempo
    SIEMPRE en la última columna (np.hstack((X_sub, horas))). Aquí fijamos la
    composición que ha introducido el usuario y barremos el tiempo.
    """
    modelo   = paquete['modelo_unico']
    features = paquete['features_ok']
    imputer  = paquete.get('imputer')

    # Fila de features del usuario, en el MISMO orden que en entrenamiento
    fila = np.array([[valores.get(f, 0.0) for f in features]], dtype=float)
    if imputer is not None:
        # No-op si la fila está completa; reproduce el preprocesado del entrenamiento
        fila = imputer.transform(fila)

    # Matriz [features..., tiempo] para cada instante del eje
    X = np.column_stack([np.tile(fila, (len(horas_eje), 1)), horas_eje])
    preds = np.clip(modelo.predict(X), 0, None)
    return np.asarray(horas_eje, dtype=float), preds


def curva_suave(horas, preds, monotona=True, n_anclas=25):
    """
    Suaviza la curva escalonada del modelo. PCHIP es un INTERPOLANTE: pasa por
    todos los puntos que recibe, así que si le pasáramos los 300 del barrido
    calcaría la escalera. La clave es submuestrear a unos pocos NODOS (n_anclas)
    y dejar que PCHIP trace una curva limpia entre ellos.

    Si monotona=True se aplica regresión isotónica para que la resistencia nunca
    decrezca con el tiempo (físicamente correcto: el hormigón no se ablanda).
    """
    orden_idx = np.argsort(horas)
    h, p = horas[orden_idx], preds[orden_idx]
    # Deduplicar horas repetidas
    h_unicas = np.unique(h)
    p_unicas = np.array([p[h == hu].mean() for hu in h_unicas])

    # Forzar monotonía creciente sobre el tiempo (en escala log)
    if monotona:
        iso = IsotonicRegression(increasing=True)
        p_unicas = iso.fit_transform(np.log(h_unicas), p_unicas)

    # Submuestrear nodos (uniforme en escala log) ANTES de interpolar
    n_anclas = int(np.clip(n_anclas, 4, len(h_unicas)))
    idx = np.linspace(0, len(h_unicas) - 1, n_anclas, dtype=int)
    h_anc, p_anc = h_unicas[idx], p_unicas[idx]
    if monotona:
        p_anc = np.maximum.accumulate(p_anc)   # blindar monotonía tras submuestrear

    interp = PchipInterpolator(h_anc, p_anc)
    h_fino = np.logspace(np.log10(h_unicas.min()), np.log10(h_unicas.max()), 300)
    return h_fino, np.clip(interp(h_fino), 0, None)


def etiqueta_x(h):
    if h < 1:    return f"{int(round(h*60))} min"
    elif h < 24: return f"{h:.4g} h"
    else:        return f"{h/24:.4g} d"


HITOS = [(24, "1 día"), (168, "7 días"), (672, "28 días")]

def evaluar_en(hx, px, objetivo):
    """Interpola la curva MOSTRADA en un tiempo objetivo (en escala log)."""
    if hx is None:
        return None
    return float(np.interp(np.log(objetivo), np.log(hx), px))


fig = go.Figure()
curva_rf = curva_xgb = (None, None)

def añadir_modelo(paquete, color, nombre, dash, symbol):
    horas, preds = predecir(paquete)   # preds = escalones crudos (salida directa del modelo)

    # Curva procesada (suavizada / monótona) que se muestra como principal
    if suavizar:
        hx, px = curva_suave(horas, preds, monotona=monotona, n_anclas=N_ANCLAS)
    elif monotona:
        # Sin suavizar pero forzando monotonía: isotónica sobre los escalones crudos
        hx, px = horas, IsotonicRegression(increasing=True).fit_transform(np.log(horas), preds)
        px = np.clip(px, 0, None)
    else:
        hx, px = horas, preds  # escalones crudos, tal cual predice el modelo

    # Escalones crudos como referencia ("valor verdadero" del modelo), solo cuando
    # hay un procesado que comparar (suavizado o monotonía activos)
    if mostrar_crudo and (suavizar or monotona):
        fig.add_trace(go.Scatter(
            x=horas, y=preds, mode='lines', name=f'{nombre} (crudo)',
            line=dict(color=color, width=1.3, shape='hv'), opacity=0.45,
            hovertemplate=f'{nombre} crudo<br>Tiempo: %{{x:.2f}} h<br>'
                          'Resistencia: %{y:.1f} MPa<extra></extra>'))

    # Curva principal procesada
    fig.add_trace(go.Scatter(
        x=hx, y=px, mode='lines', name=nombre,
        line=dict(color=color, width=3, dash=dash),
        hovertemplate=f'{nombre}<br>Tiempo: %{{x:.2f}} h<br>Resistencia: %{{y:.1f}} MPa<extra></extra>'))

    # Marcadores en edades clave (1 d, 7 d, 28 d)
    hitos_h = [h for h, _ in HITOS]
    y_hitos = [evaluar_en(hx, px, h) for h in hitos_h]
    fig.add_trace(go.Scatter(
        x=hitos_h, y=y_hitos, mode='markers', name=f'{nombre} (edades clave)',
        showlegend=False,
        marker=dict(color=color, size=10, symbol=symbol, line=dict(color='white', width=1.5)),
        hovertemplate=f'{nombre}<br>%{{x:.0f}} h<br>%{{y:.1f}} MPa<extra></extra>'))
    return hx, px

if mostrar_rf:
    curva_rf = añadir_modelo(paquete_rf, '#2563EB', 'Random Forest', None, 'circle')
if mostrar_xgb:
    curva_xgb = añadir_modelo(paquete_xgb, '#DC2626', 'XGBoost', 'dash', 'square')

# Ejes en edades "redondas" e interpretables
ticks = [3/60, 1, 4, 24, 72, 168, 672]
fig.update_xaxes(tickvals=ticks, ticktext=[etiqueta_x(t) for t in ticks])

fig.update_layout(
    xaxis_type='log',
    xaxis_title='Tiempo de curado (escala logarítmica)',
    yaxis_title='Resistencia estimada (MPa)',
    hovermode='x unified',
    plot_bgcolor='white', paper_bgcolor='#F8FAFC',
    font=dict(color='#334155'),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    margin=dict(l=60, r=30, t=40, b=60), height=480,
)
fig.update_xaxes(gridcolor='#E2E8F0', showline=True, linecolor='#CBD5E1')
fig.update_yaxes(gridcolor='#E2E8F0', showline=True, linecolor='#CBD5E1')

st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 7. DASHBOARD A EDADES CLAVE
# ═══════════════════════════════════════════════════════════════════════════
st.subheader("Comparativa de Predicciones a Edades Clave")

def fmt(hx, px, objetivo):
    v = evaluar_en(hx, px, objetivo)
    return f"{v:.1f} MPa" if v is not None else "N/A"

fila_cab = st.columns([1, 2, 2, 2])
fila_cab[0].markdown("&nbsp;", unsafe_allow_html=True)
for (h_t, nombre), c in zip(HITOS, fila_cab[1:]):
    c.markdown(f"**{nombre}**  ·  {h_t} h")

if mostrar_rf:
    fila = st.columns([1, 2, 2, 2])
    fila[0].markdown('<div class="fila-modelo">Random Forest</div>', unsafe_allow_html=True)
    for (h_t, _), c in zip(HITOS, fila[1:]):
        c.metric("", fmt(curva_rf[0], curva_rf[1], h_t))

if mostrar_xgb:
    fila = st.columns([1, 2, 2, 2])
    fila[0].markdown('<div class="fila-modelo">XGBoost</div>', unsafe_allow_html=True)
    for (h_t, _), c in zip(HITOS, fila[1:]):
        c.metric("", fmt(curva_xgb[0], curva_xgb[1], h_t))