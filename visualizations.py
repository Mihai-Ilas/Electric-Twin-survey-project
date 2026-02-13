import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

def plot_clusters_3d(questions, embeddings, labels):
    # Reduce the 768-dimensional embeddings to 3 dimensions
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(embeddings.cpu())
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d') # Changed '100' to '111' (standard subplot syntax)
    
    ax.scatter(
        reduced_data[:, 0], 
        reduced_data[:, 1], 
        reduced_data[:, 2], 
        c=labels, 
        cmap='tab20', 
        s=60, 
        alpha=0.8, 
        edgecolors='k', 
        linewidth=0.5
    )
    
    ax.set_title("3D Projection of Survey Latent Structure")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    
    plt.tight_layout()
    
    output_path = Path("visualizations")
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path / "cluster_visualization.png", dpi=300, bbox_inches='tight')