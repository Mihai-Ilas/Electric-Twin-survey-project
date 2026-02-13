import pickle
import model
import evaluation
import visualizations
import data_loading
from config import INPUT_DATA_FILE, GROUND_THRUTHS_FILE, ENCODING_OUTPUTS_FILE

if __name__ == "__main__":
    
    try:
        # 1. Load data
        qs = list(set(data_loading.load_data(INPUT_DATA_FILE)))
        gt_dict = data_loading.load_ground_truth(GROUND_THRUTHS_FILE)
        
        # 2. Build model
        embeds, tfidf, tfidf_matrix = model.build_latent_model(qs)
        
        # 3. Perform clustering
        cluster_labels, cluster_metadata = model.perform_clustering(qs, embeds, tfidf, tfidf_matrix)

        # 4. Save embeddings and clusters data for later inference
        with open(ENCODING_OUTPUTS_FILE, 'wb') as f:
            pickle.dump({
                'questions': qs,
                'embeddings': embeds,
                'labels': cluster_labels,
                'cluster_metadata': cluster_metadata
            }, f)

        # 5. Execute evaluation & visualization
        evaluation.evaluate_and_log_errors(qs, cluster_labels, gt_dict, tfidf, tfidf_matrix)
        visualizations.plot_clusters_3d(qs, embeds, cluster_labels)
        
    except Exception as e:
        print(f"An error occurred: {e}")