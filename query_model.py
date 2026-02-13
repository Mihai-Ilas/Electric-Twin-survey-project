import sys
import pickle
import json
import torch
from typing import List
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from config import MODEL_NAME, INDIVIDUAL_SIMILARITY_THRESHOLD, CLUSTER_SIMILARITY_THRESHOLD, ENCODING_OUTPUTS_FILE

class ClusterContext(BaseModel):
    cluster_id: int
    questions_in_cluster: int
    top_thematic_keywords: List[str]

class AnalysisResponse(BaseModel):
    input_question: str
    closest_individual_match: str
    individual_similarity_score: float
    cluster_centroid_similarity: float
    cluster_context: ClusterContext
    outcome: str

MODEL = None
DATA = None

def load_resources():
    global MODEL, DATA
    try:
        with open(ENCODING_OUTPUTS_FILE, 'rb') as f:
            DATA = pickle.load(f)
        MODEL = SentenceTransformer(MODEL_NAME)
    except FileNotFoundError:
        print("Error: encoding outputs file not found. Run App.py first.")
        sys.exit(1)

def get_semantic_analysis(new_question: str):
    # Encode new question
    new_emb = MODEL.encode(new_question, convert_to_tensor=True)
    
    # Compute cosine similarity to closest matching question
    cos_scores = util.cos_sim(new_emb, DATA['embeddings'])[0]
    best_idx = cos_scores.argmax().item()
    indiv_sim = float(cos_scores[best_idx])
    closest_q = DATA['questions'][best_idx]
    
    # Find cluster the question would fall into
    cluster_id = int(DATA['labels'][best_idx])
    
    # Filter embeddings belonging to this specific cluster
    cluster_mask = [i for i, lbl in enumerate(DATA['labels']) if lbl == cluster_id]
    cluster_embeddings = DATA['embeddings'][cluster_mask]
    
    # Compute the average vector (Centroid) for the cluster and cosine similarity to centroid
    centroid = torch.mean(cluster_embeddings, dim=0, keepdim=True)
    centroid_sim = util.cos_sim(new_emb, centroid).item()
    
    # Get cluster metadata
    meta = DATA.get('cluster_metadata', {}).get(cluster_id, {
        "keywords": ["unknown"], 
        "size": 0
    })
    
    # We use individual similarity for redundancy and cluster centroid similarity for thematic placement.
    if indiv_sim > INDIVIDUAL_SIMILARITY_THRESHOLD:
        outcome = "REJECT: Semantically redundant with an existing question."
    elif centroid_sim > CLUSTER_SIMILARITY_THRESHOLD:
        outcome = "ACCEPT: Fits clearly within the current thematic cluster."
    else:
        outcome = "ACCEPT: Unique contribution; consider as a new sub-topic."

    return {
        "input_question": new_question,
        "closest_individual_match": closest_q,
        "individual_similarity_score": round(indiv_sim, 3),
        "cluster_centroid_similarity": round(centroid_sim, 3),
        "cluster_context": {
            "cluster_id": cluster_id,
            "questions_in_cluster": meta["size"],
            "top_thematic_keywords": [k.upper() for k in meta["keywords"]]
        },
        "outcome": outcome
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Capture and clean input
        user_input = " ".join(sys.argv[1:]).strip()
        
        # Remove accidental leading/trailing quotes
        # This fixes: "What do you think of electric cars? or 'What do you think of electric cars?
        user_input = user_input.strip("'\"")

        # Only spaces or empty input
        if not user_input:
            print("Error: Input is empty. Please provide a survey question.")
            sys.exit(1)

        # Input is too short (e.g., just a "?" or "a")
        if len(user_input) < 5:
            print("Error: Input too short to analyze. Please provide a longer question")
            sys.exit(1)

        # Edge Case: Input is way too long
        if len(user_input) > 1000:
            print("Warning: Input is very long. Results may be less accurate.")
            user_input = user_input[:1000]

        try:
            load_resources()
            result = get_semantic_analysis(user_input)
            print(json.dumps(result, indent=2))
            
        except Exception as e:
            # Resource loading or model failure
            print(f"An unexpected error occurred: {e}")
            sys.exit(1)
    else:
        print("Please follow the format: python query_model.py 'Your survey question here'")