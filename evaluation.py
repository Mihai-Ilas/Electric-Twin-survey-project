import numpy as np
from utils import get_cluster_keywords
from config import INCORRECT_PREDICTIONS_FILE

# Mapping Ground Truth Labels to possible keyword substrings
LABEL_KEYWORDS = {
    "HEALTHCARE": ["health", "medical", "provider", "care", "diagnose", "treatment", "doctor", "specialist"],
    "FOOD": ["food", "fast", "diet", "nutrition", "taste", "restaurant", "menu", "eat", "meal"],
    "CARS": ["car", "electric", "vehicle", "charging", "battery", "driving", "range", "engine"],
    "POLITICS/COUNTRY": ["political", "government", "election", "country", "leadership", "policy"],
    "BOOKS": ["book", "books", "read", "author", "print", "literary"],
    "WORK": ["work", "job", "career", "office", "employee", "professional"],
    "OTHER": [] # If label is OTHER, it's always accurate unless it matches a specific category above
}


def evaluate_and_log_errors(questions, final_labels, gt_dict, tfidf, tfidf_matrix, error_file=INCORRECT_PREDICTIONS_FILE):    
    correct = 0
    total_found = 0
    mismatches = []

    unique_clusters = np.unique(final_labels)
    cluster_anchors = {
        cid: [kw.lower() for kw in get_cluster_keywords(final_labels, cid, tfidf, tfidf_matrix)] 
        for cid in unique_clusters if cid != -1
    }
    
    for i, q in enumerate(questions):
        q_lookup = q.strip().lower()
        if q_lookup not in gt_dict:
            continue
        
        total_found += 1
        true_tag = gt_dict[q_lookup]
        cluster_id = final_labels[i]
        
        anchors = cluster_anchors.get(cluster_id, [])
        anchor_str = ", ".join(anchors).upper() if anchors else "NOISE/NONE"
        
        is_correct = False
        
        if true_tag == "OTHER":
            if cluster_id == -1:
                # We correctly identified a question of OTHER type
                is_correct = True
            else:
                match_major_cluster = False
                for cat, syns in LABEL_KEYWORDS.items():
                    if cat == "OTHER": continue
                    if any(any(s.lower() in a for s in syns) for a in anchors):
                        match_major_cluster = True
                        break
                if not match_major_cluster:
                    # Output is not one of the major clusters, so it can be considered OTHER
                    is_correct = True
        
        elif cluster_id != -1:
            synonyms = LABEL_KEYWORDS.get(true_tag, [])
            if any(any(s.lower() in a for s in synonyms) for a in anchors):
                is_correct = True
                
        if is_correct:
            correct += 1
        else:
            mismatches.append(f"{q} - Ground truth {true_tag} - Model output {anchor_str}")

    with open(error_file, "w", encoding="utf-8") as f:
        f.write(f"INCORRECT PREDICTIONS (Total: {len(mismatches)})\n")
        f.write("\n".join(mismatches))

    accuracy = (correct / total_found) * 100 if total_found > 0 else 0
    print(f">> Accuracy: {accuracy:.2f}%")
    return accuracy