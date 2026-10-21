import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN

# ---------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard 3D - Manutenção Preditiva",
    page_icon="⚡",
    layout="wide"
)

# Estilização CSS Personalizada (Cores Chamativas / Dark Theme)
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #f0f6fc;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Dashboard 3D de Manutenção Preditiva (DBSCAN)")
st.markdown("Detecção em tempo real de anomalias industriais com visualização tridimensional e cores de alto contraste.")

# ---------------------------------------------------------
# 1. GERAÇÃO DOS DADOS SINTÉTICOS (NumPy + Pandas)
# ---------------------------------------------------------
@st.cache_data
def generate_sensor_data(n_samples=350, random_state=42):
    np.random.seed(random_state)
    
    # Cluster 1: Operação Normal A
    c1_vib = np.random.normal(loc=45, scale=4, size=int(n_samples * 0.42))
    c1_temp = np.random.normal(loc=60, scale=3.5, size=int(n_samples * 0.42))
    
    # Cluster 2: Operação Normal B (Alta Carga)
    c2_vib = np.random.normal(loc=75, scale=5, size=int(n_samples * 0.42))
    c2_temp = np.random.normal(loc=82, scale=4, size=int(n_samples * 0.42))
    
    # Outliers: Falhas/Anomalias
    n_outliers = n_samples - len(c1_vib) - len(c2_vib)
    outliers_vib = np.random.uniform(low=20, high=110, size=n_outliers)
    outliers_temp = np.random.uniform(low=35, high=110, size=n_outliers)
    
    # Unificação dos dados
    vibracao = np.concatenate([c1_vib, c2_vib, outliers_vib])
    temperatura = np.concatenate([c1_temp, c2_temp, outliers_temp])
    
    # Adicionando uma 3ª dimensão calculada: Índice de Severidade / Carga
    severidade = (vibracao * 0.6) + (temperatura * 0.4) + np.random.normal(0, 3, size=n_samples)
    
    machine_ids = [f"MAQ-{i+1:03d}" for i in range(n_samples)]
    
    return pd.DataFrame({
        'ID_Maquina': machine_ids,
        'Frequencia_Vibracao': np.round(vibracao, 2),
        'Pico_Temperatura': np.round(temperatura, 2),
        'Indice_Severidade': np.round(severidade, 2)
    })

df_sensors = generate_sensor_data()

# ---------------------------------------------------------
# 2. PAINEL LATERAL DE PARÂMETROS
# ---------------------------------------------------------
st.sidebar.header("🎛️ Parâmetros do DBSCAN")
st.sidebar.markdown("Ajuste a sensibilidade da detecção:")

eps_param = st.sidebar.slider(
    "Distância Máxima (`eps`):",
    min_value=1.0,
    max_value=15.0,
    value=6.0,
    step=0.5
)

min_samples_param = st.sidebar.slider(
    "Mínimo de Amostras (`min_samples`):",
    min_value=2,
    max_value=20,
    value=5,
    step=1
)

# ---------------------------------------------------------
# 3. ALGORITMO DBSCAN
# ---------------------------------------------------------
X = df_sensors[['Frequencia_Vibracao', 'Pico_Temperatura', 'Indice_Severidade']].values

dbscan = DBSCAN(eps=eps_param, min_samples=min_samples_param)
df_sensors['Cluster'] = dbscan.fit_predict(X)

# Definição de Status
df_sensors['Status'] = df_sensors['Cluster'].apply(
    lambda x: '🚨 Anomalia (Outlier)' if x == -1 else f'🟢 Normal (Grupo {x})'
)
df_sensors['Tipo'] = df_sensors['Cluster'].apply(
    lambda x: 'Anomalia' if x == -1 else 'Normal'
)

n_clusters = len(set(df_sensors['Cluster'])) - (1 if -1 in df_sensors['Cluster'].values else 0)
n_anomalies = (df_sensors['Cluster'] == -1).sum()

# Métrica Resumo
m1, m2, m3 = st.columns(3)
m1.metric("Total de Máquinas", len(df_sensors))
m2.metric("Grupos Normais", n_clusters)
m3.metric("Anomalias Detectadas", n_anomalies)

# ---------------------------------------------------------
# 4. GRÁFICO 3D COM CORES CHAMATIVAS E SEM LEGENDA
# ---------------------------------------------------------
st.subheader("🧊 Modelo 3D de Vibração x Temperatura x Severidade")

# Atribuindo cores vibrantes chamativas para cada grupo
# Outliers (-1) ficam em Rosa/Magenta Neon
