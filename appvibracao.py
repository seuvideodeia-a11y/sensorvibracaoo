import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

# Configuração da página Streamlit
st.set_page_config(
    page_title="Dashboard de Manutenção Preditiva",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Dashboard de Manutenção Preditiva com DBSCAN")
st.markdown("""
Este dashboard utiliza dados simula dos de sensores de chão de fábrica (**Frequência de Vibração** e **Pico de Temperatura**) 
para identificar padrões de operação normal e detectar **anomalias/outliers** via agrupamento espacial baseado em densidade (DBSCAN).
""")

# ---------------------------------------------------------
# 1. GERAÇÃO DOS DADOS (NumPy + Pandas)
# ---------------------------------------------------------
@st.cache_data
def generate_sensor_data(n_samples=300, random_state=42):
    np.random.seed(random_state)
    
    # Cluster 1: Funcionamento Normal (Padrão A)
    cluster1_vib = np.random.normal(loc=50, scale=5, size=int(n_samples * 0.45))
    cluster1_temp = np.random.normal(loc=65, scale=4, size=int(n_samples * 0.45))
    
    # Cluster 2: Funcionamento Normal (Padrão B / Alta Carga)
    cluster2_vib = np.random.normal(loc=80, scale=6, size=int(n_samples * 0.45))
    cluster2_temp = np.random.normal(loc=85, scale=5, size=int(n_samples * 0.45))
    
    # Outliers: Comportamentos Anômalos
    n_outliers = n_samples - len(cluster1_vib) - len(cluster2_vib)
    outliers_vib = np.random.uniform(low=20, high=110, size=n_outliers)
    outliers_temp = np.random.uniform(low=40, high=110, size=n_outliers)
    
    # Consolidação dos dados
    vibracao = np.concatenate([cluster1_vib, cluster2_vib, outliers_vib])
    temperatura = np.concatenate([cluster1_temp, cluster2_temp, outliers_temp])
    
    machine_ids = [f"MÁQ-{i+1:03d}" for i in range(n_samples)]
    
    df = pd.DataFrame({
        'ID_Maquina': machine_ids,
        'Frequencia_Vibracao': vibracao,
        'Pico_Temperatura': temperatura
    })
    
    return df

df_sensors = generate_sensor_data()

# ---------------------------------------------------------
# 2. BARRA LATERAL (Controles do DBSCAN)
# ---------------------------------------------------------
st.sidebar.header("🔧 Parâmetros do DBSCAN")
st.sidebar.write("Ajuste a sensibilidade do algoritmo para detecção de clusters e anomalias:")

eps_param = st.sidebar.slider(
    "Distância Máxima (`eps`):",
    min_value=1.0,
    max_value=15.0,
    value=5.5,
    step=0.5,
    help="Distância máxima entre dois pontos para que sejam considerados vizinhos."
)

min_samples_param = st.sidebar.slider(
    "Mínimo de Amostras (`min_samples`):",
    min_value=2,
    max_value=20,
    value=6,
    step=1,
    help="Número mínimo de pontos em uma vizinhança para formar uma região densa."
)

# ---------------------------------------------------------
# 3. APLICAÇÃO DO ALGORITMO DBSCAN
# ---------------------------------------------------------
X = df_sensors[['Frequencia_Vibracao', 'Pico_Temperatura']].values

dbscan = DBSCAN(eps=eps_param, min_samples=min_samples_param)
df_sensors['Cluster'] = dbscan.fit_predict(X)

# Identificação de anomalias (DBSCAN rotula outliers/ruído como -1)
df_sensors['Status'] = df_sensors['Cluster'].apply(lambda x: 'Anomalia (Outlier)' if x == -1 else f'Cluster {x}')

n_clusters = len(set(df_sensors['Cluster'])) - (1 if -1 in df_sensors['Cluster'].values else 0)
n_anomalies = (df_sensors['Cluster'] == -1).sum()

# Métricas rápidas no topo
col1, col2, col3 = st.columns(3)
col1.metric("Total de Máquinas Monitoradas", len(df_sensors))
col2.metric("Clusters Normais Detectados", n_clusters)
col3.metric("Anomalias Identificadas", n_anomalies, delta_color="inverse")

# ---------------------------------------------------------
# 4. VISUALIZAÇÃO GRÁFICA (Seaborn / Matplotlib)
# ---------------------------------------------------------
st.subheader("📊 Agrupamento de Sensores e Detecção de Outliers")

fig, ax = plt.subplots(figsize=(10, 6))

# Define paleta de cores destacando outliers (-1) em preto
unique_clusters = sorted(df_sensors['Cluster'].unique())
palette = {}

for c in unique_clusters:
    if c == -1:
        palette[c] = "#000000"  # Preto para outliers
    else:
        # Paleta colorida para clusters normais
        palette[c] = sns.color_palette("tab10")[c % 10]

# Gráfico de dispersão
sns.scatterplot(
    data=df_sensors,
    x='Frequencia_Vibracao',
    y='Pico_Temperatura',
    hue='Cluster',
    palette=palette,
    style=(df_sensors['Cluster'] == -1),  # Muda o formato dos pontos de outlier
    markers={False: 'o', True: 'X'},
    s=90,
    alpha=0.8,
    ax=ax
)

# Estilização do gráfico
ax.set_title("Frequência de Vibração vs. Pico de Temperatura", fontsize=14, pad=15)
ax.set_xlabel("Frequência de Vibração (Hz)", fontsize=12)
ax.set_ylabel("Pico de Temperatura (°C)", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)

# Ajuste da legenda
handles, labels = ax.get_legend_handles_labels()
new_labels = ["Outlier (Anomalia)" if l == "-1" else f"Cluster {l}" for l in labels]
ax.legend(handles=handles, labels=new_labels, title="Agrupamentos", bbox_to_anchor=(1.02, 1), loc='upper left')

st.pyplot(fig)

# ---------------------------------------------------------
# 5. TABELA INTERATIVA DE MÁQUINAS ANÔMALAS
# ---------------------------------------------------------
st.subheader("🚨 Máquinas com Comportamento Anômalo")

df_anomalies = df_sensors[df_sensors['Cluster'] == -1][['ID_Maquina', 'Frequencia_Vibracao', 'Pico_Temperatura']].reset_index(drop=True)

if not df_anomalies.empty:
    st.warning(f"Atenção: Foram encontradas **{len(df_anomalies)}** máquinas operando fora dos parâmetros normais.")
    st.dataframe(
        df_anomalies.style.format({
            'Frequencia_Vibracao': '{:.2f} Hz',
            'Pico_Temperatura': '{:.2f} °C'
        }),
        use_container_width=True
    )
else:
    st.success("Nenhuma anomalia detectada com os parâmetros atuais de DBSCAN.")
