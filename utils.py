import numpy as np

def get_cluster_keywords(labels_array, target_cluster, tfidf, tfidf_matrix):
    feature_names = tfidf.get_feature_names_out()
    global_mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    indices = np.where(labels_array == target_cluster)[0]
    if len(indices) == 0: return set()
    c_mean = np.asarray(tfidf_matrix[indices].mean(axis=0)).flatten()
    keyness = c_mean - global_mean_tfidf 
    top_indices = keyness.argsort()[-5:][::-1]
    return set([feature_names[idx] for idx in top_indices])