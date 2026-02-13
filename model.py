import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist
from config import MODEL_NAME, NOISE_CENTRALITY_THRESHOLD, CLUSTERING_RESULTS_FILE
from utils import get_cluster_keywords

def build_latent_model(questions):
    print("Encoding the semantic space...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(questions, convert_to_tensor=True)
    
    # TF-IDF to find "Signature Words" by de-emphasizing common survey phrases
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7) 
    tfidf_matrix = tfidf.fit_transform(questions)
    
    return embeddings, tfidf, tfidf_matrix

def perform_clustering(questions, embeddings, tfidf, tfidf_matrix):
    # Normalize embeddings and find optimal number of clusters
    embeds_np = normalize(embeddings.cpu().numpy()) 
    best_k, best_score = 3, -1
    for k in range(3, 11):
        test_kmeans = KMeans(n_clusters=k, random_state=99, n_init=10)
        test_labels = test_kmeans.fit_predict(embeds_np)
        score = silhouette_score(embeds_np, test_labels)
        if score > best_score:
            best_score, best_k = score, k
            
    # Initial KMeans clustering with optional k
    kmeans = KMeans(n_clusters=best_k, random_state=99, n_init=10)
    initial_labels = kmeans.fit_predict(embeds_np)

    # Merge clusters which have an overlap in anchor keywords
    cluster_keys = {c: get_cluster_keywords(initial_labels, c, tfidf, tfidf_matrix) for c in range(best_k)}
    merge_map = {c: c for c in range(best_k)}
    for c1 in range(best_k):
        for c2 in range(c1 + 1, best_k):
            if cluster_keys[c1].intersection(cluster_keys[c2]) and len(cluster_keys[c1].intersection(cluster_keys[c2])) >= 2:
                merge_map[c2] = merge_map[c1]

    consolidated_labels = np.array([merge_map[l] for l in initial_labels])

    current_unique_ids = sorted(list(set(consolidated_labels)))
    centroid_list = []
    final_unique_ids = []

    for cid in current_unique_ids:
        mask = (consolidated_labels == cid)
        if np.any(mask):
            centroid_list.append(embeds_np[mask].mean(axis=0))
            final_unique_ids.append(cid)

    distances = cdist(embeds_np, centroid_list, 'euclidean')
    
    # Any point which has a centrality of less than a threshold to a cluster is marked as OTHER
    threshold = NOISE_CENTRALITY_THRESHOLD
    final_labels = []
    for i, old_label in enumerate(consolidated_labels):
        # We index based on the position in final_unique_ids
        try:
            c_idx = final_unique_ids.index(old_label)
            centrality = 1 - distances[i, c_idx]
            final_labels.append(old_label if centrality >= threshold else -1)
        except ValueError:
            final_labels.append(-1)
    
    final_labels = np.array(final_labels)

    # Metadata for API response
    cluster_metadata = {}
    for cluster_id in final_unique_ids:
        indices = np.where(final_labels == cluster_id)[0]
        if len(indices) == 0: continue
        keys = get_cluster_keywords(final_labels, cluster_id, tfidf, tfidf_matrix)
        cluster_metadata[int(cluster_id)] = {"keywords": list(keys), "size": len(indices)}
    
    noise_indices = np.where(final_labels == -1)[0]
    cluster_metadata[-1] = {"keywords": ["UNGROUPED"], "size": len(noise_indices)}

    # Generate clustering report
    with open(CLUSTERING_RESULTS_FILE, "w", encoding="utf-8") as f:
        for idx, cluster_id in enumerate(final_unique_ids):

            indices = np.where(final_labels == cluster_id)[0]
            if len(indices) == 0: continue

            keys = cluster_metadata[cluster_id]["keywords"]
            f.write(f"CLUSTER {idx+1}\nTHEMATIC ANCHORS: {', '.join(keys).upper()}\n" + "="*30 + "\n")
            
            members = []
            for i in indices:
                # Use the index in final_unique_ids to get distance
                members.append((questions[i], 1 - distances[i, final_unique_ids.index(cluster_id)]))
            members.sort(key=lambda x: x[1], reverse=True)
            for r, (q, c) in enumerate(members, 1):
                f.write(f"{r}. [Centrality: {c:.4f}] {q}\n")
            f.write("\n\n")
        
        if len(noise_indices) > 0:
            f.write("OUTLIERS / NOISE CLUSTER (-1)\nTHEMATIC ANCHORS: NONE\n" + "="*30 + "\n")
            for i, idx in enumerate(noise_indices, 1):
                f.write(f"{i}. {questions[idx]}\n")

    return final_labels, cluster_metadata