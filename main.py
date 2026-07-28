import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==========================================
# CONFIG & JUDUL APLIKASI
# ==========================================
st.set_page_config(
    page_title="Optimasi Segmentasi Pelanggan - PSO K-Means",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Segmentasi Pelanggan E-Commerce")
st.caption("Optimasi Inisialisasi Centroid K-Means Menggunakan Algoritma Particle Swarm Optimization (PSO)")
st.markdown("---")

# ==========================================
# CLASS PSO FOR K-MEANS
# ==========================================
class PSOKMeans:
    def __init__(self, n_clusters, n_particles=30, max_iter=50):
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.max_iter = max_iter

    def _compute_inertia(self, centroids, data):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        min_distances = np.min(distances, axis=1)
        return np.sum(min_distances ** 2)

    def fit(self, data):
        n_samples, n_features = data.shape
        min_val, max_val = np.min(data, axis=0), np.max(data, axis=0)
        
        particles = np.random.uniform(min_val, max_val, (self.n_particles, self.n_clusters, n_features))
        velocities = np.zeros_like(particles)
        
        pbest_positions = np.copy(particles)
        pbest_scores = np.array([self._compute_inertia(p, data) for p in particles])
        
        gbest_index = np.argmin(pbest_scores)
        gbest_position = np.copy(pbest_positions[gbest_index])
        gbest_score = pbest_scores[gbest_index]
        
        w, c1, c2 = 0.5, 1.5, 1.5
        
        for _ in range(self.max_iter):
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (w * velocities[i] + 
                                 c1 * r1 * (pbest_positions[i] - particles[i]) + 
                                 c2 * r2 * (gbest_position - particles[i]))
                particles[i] += velocities[i]
                
                current_score = self._compute_inertia(particles[i], data)
                if current_score < pbest_scores[i]:
                    pbest_scores[i] = current_score
                    pbest_positions[i] = np.copy(particles[i])
                    
                    if current_score < gbest_score:
                        gbest_score = current_score
                        gbest_position = np.copy(particles[i])
                        
        return gbest_position

# ==========================================
# LOAD DATASET (WITH FALLBACK)
# ==========================================
@st.cache_data
def load_data():
    # URL Utama
    url = "https://raw.githubusercontent.com/SteffiPeT4/MachineLearning/master/Mall_Customers.csv"
    try:
        return pd.read_csv(url)
    except Exception:
        # Fallback URL jika URL pertama bermasalah/404
        alt_url = "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"
        return pd.read_csv(alt_url)

try:
    df = load_data()
except Exception as e:
    st.error("Gagal mengunduh dataset secara otomatis. Silakan periksa koneksi internet Anda.")
    st.stop()

# ==========================================
# SIDEBAR - CONTROL PANEL
# ==========================================
st.sidebar.header("⚙️ Pengaturan Parameter")

# Parameter Clustering & PSO
n_clusters = st.sidebar.slider("Jumlah Klaster (K)", min_value=2, max_value=8, value=5)
n_particles = st.sidebar.slider("Jumlah Partikel (PSO)", min_value=10, max_value=50, value=30, step=5)
max_iter = st.sidebar.slider("Maksimal Iterasi (PSO)", min_value=10, max_value=100, value=50, step=10)

btn_run = st.sidebar.button("🚀 Jalankan Optimasi", type="primary")

# ==========================================
# MAIN PAGE - DATA & PROCESSING
# ==========================================
st.subheader("📊 Dataset: Mall Customers")
col_data, col_stats = st.columns([2, 1])

with col_data:
    st.dataframe(df.head(6), use_container_width=True)

with col_stats:
    st.metric("Total Pelanggan", len(df))
    st.metric("Fitur Digunakan", "Annual Income & Spending Score")

# Preprocessing Data
X = df.iloc[:, [3, 4]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if btn_run or "executed" in st.session_state:
    st.session_state["executed"] = True
    
    with st.spinner("Mengoptimasi Centroid dengan PSO... Mohon tunggu..."):
        # 1. K-Means Standar
        kmeans_std = KMeans(n_clusters=n_clusters, init='random', random_state=42, n_init=10)
        labels_std = kmeans_std.fit_predict(X_scaled)
        inertia_std = kmeans_std.inertia_
        sil_std = silhouette_score(X_scaled, labels_std)

        # 2. K-Means + PSO
        pso = PSOKMeans(n_clusters=n_clusters, n_particles=n_particles, max_iter=max_iter)
        pso_centroids = pso.fit(X_scaled)
        
        kmeans_pso = KMeans(n_clusters=n_clusters, init=pso_centroids, n_init=1, random_state=42)
        labels_pso = kmeans_pso.fit_predict(X_scaled)
        inertia_pso = kmeans_pso.inertia_
        sil_pso = silhouette_score(X_scaled, labels_pso)

    st.markdown("---")
    st.subheader("📈 Hasil Evaluasi Perbandingan")
    
    # KARTU METRIK EVALUASI
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Inertia (Standard)", f"{inertia_std:.2f}")
    m2.metric("Inertia (PSO Optimized)", f"{inertia_pso:.2f}", 
              delta=f"{inertia_pso - inertia_std:.2f}", delta_color="inverse")
    
    m3.metric("Silhouette (Standard)", f"{sil_std:.4f}")
    m4.metric("Silhouette (PSO Optimized)", f"{sil_pso:.4f}", 
              delta=f"{sil_pso - sil_std:.4f}")

    # VISUALISASI KLASTER
    st.markdown("### 🎨 Visualisasi Klaster Pelanggan")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Standar
    axes[0].scatter(X[:, 0], X[:, 1], c=labels_std, cmap='viridis', alpha=0.6, edgecolors='k')
    axes[0].set_title(f"K-Means Standar\nInertia: {inertia_std:.1f} | Silhouette: {sil_std:.3f}")
    axes[0].set_xlabel("Annual Income (k$)")
    axes[0].set_ylabel("Spending Score (1-100)")

    # Plot PSO
    axes[1].scatter(X[:, 0], X[:, 1], c=labels_pso, cmap='viridis', alpha=0.6, edgecolors='k')
    
    pso_centers_original = scaler.inverse_transform(kmeans_pso.cluster_centers_)
    axes[1].scatter(pso_centers_original[:, 0], pso_centers_original[:, 1], 
                    s=200, c='red', marker='X', label='Centroid PSO')
    axes[1].set_title(f"K-Means + PSO Optimasi\nInertia: {inertia_pso:.1f} | Silhouette: {sil_pso:.3f}")
    axes[1].set_xlabel("Annual Income (k$)")
    axes[1].set_ylabel("Spending Score (1-100)")
    axes[1].legend()

    st.pyplot(fig)

    # RINGKASAN SEGMEN
    st.markdown("### 📝 Profil Klaster Pelanggan (Hasil PSO)")
    df['Cluster'] = labels_pso
    cluster_summary = df.groupby('Cluster').agg({
        'Annual Income (k$)': 'mean',
        'Spending Score (1-100)': 'mean',
        'CustomerID': 'count'
    }).rename(columns={'CustomerID': 'Jumlah Pelanggan'})
    
    st.dataframe(cluster_summary.style.highlight_max(axis=0), use_container_width=True)

else:
    st.info("👈 Silakan atur parameter di sidebar lalu klik tombol **Jalankan Optimasi** untuk melihat hasil.")