# Usage:

In a terminal:

Install dependencies:

**pip install -r requirements.txt**

Navigate to the package location.

Generate model (run only once):

**python generate\_model.py**

Query model:

**python query\_model.py "Your question here?"**

The "Artifacts" section contains more details about files containing interesting output information.


# Solution description:
In order to compute similarity scores between questions, we need to model their meaning. While using an LLM for pair-wise comparison is the easiest method, this does not scale well to comparisons against a large dataset. Instead, we are generating semantic embeddings for each question, using an open-source base model. 'all-mpnet-base-v2' has a good trade-off bettween performance and memory size.

Examining the data, we see that the questions follow a few major thematic groups: Politics, electric cars etc.

We are modelling this by clustering the embeddings, using the KMeans algorithm. The challenge with custering algorithm is that it depends on a fixed cluster number k.
Instead, we are computing the optimal k by looping over values in a sensible interval (in this case 3-11) and picking the value maximizing the silhouette score. This is a metric which takes into account both the cohesion and separation (unlike the Elbow method for example).

For any question which is very far from the center of its assigned cluster, we categorize it as a weak similarity and instead bundle it into an OTHER cluster.

Another challenge with this approach is that a lot of questions can be similar in two different ways:

1. The beginning: e.g. "How often do you ....", "What do you think of...."
2. The actual subject of the question: e.g. "... celebrities", "... electric cars"

Consider the question:
"How often do you read books?"

Which of the following questions is more similar?
A: "What is your opinion on books?"
B: "How often do you drive"

For the purpose of our survey, the question A is more similar, however the cosine similarity with the embedding of B may be comparable since both question B and the target one share the theme of "Frequency of an action", even though this is not very important in our context.

Ideally, we would want to bias towards the end of the question, but not fully (the first half contains information about the type of the question, which is still relevant).

While we cannot control the embeddings directly, one indirect way to tackle this is to use TF-IDF (Term Frequency-Inverse Document Frequency) to understand the importance of words in a question. This takes into account 2 dimensions:

1. How often a word appears in each sentence.
2. How often a word appears throughout the document.

The result is that words which are very common across questions spanning multiple themes (such as "How", "What" etc.) receive a lower weight. This makes sense intuitively, since a question from the dataset is a best match when it is most similar to the target question in the more "unique" parts.

We use the resulting tf-idf matrix when assigning the Thematic keywords for each cluster (for each cluster we generate 5 of these Anchors which define the commonality of the questions in the cluster).
When 2 clusters have at least 2/5 anchors in common, it means the shape of the question was different, but the theme was actualy similar, so we merge the clusters and re-compute the centroid.

Finally, we expose an API where the user can query any question. We find both the most similar question in the dataset to the target one, and the cluster the target would correspond to. We respond in the following format:
{
"input\_question": "What do you think of electric cars?",
"closest\_individual\_match": "What are your thoughts on the aesthetics and design of electric cars?",
"individual\_similarity\_score": 0.874, -> cosine score with the most similar question in the survey
"cluster\_centroid\_similarity": 0.85, -> cosine similarity with the centroid of the cluster the question would belong to
"cluster\_context": {
"cluster\_id": 1,
"questions\_in\_cluster": 91,
"top\_thematic\_keywords": \[
"CARS",
"ELECTRIC",
"IMPORTANT",
"CAR",
"THINK"
]
},
"outcome": "ACCEPT: Fits clearly within the current thematic cluster."
}



# Evaluation:
To evaluate the model, I used an LLM to assign one of the following labels to a subset of questions:
POLITICS/COUNTRY, WORK, FOOD, CELEBRITIES, CARS, HEALTHCARE, BOOKS, OTHER. This was a one-off process and the results are written to data/ground\_truth.txt (do not delete this file!).

For each of these themes, I also hardcoded a list of possible words from the same lexical field.

A clustering outputs is deemed correct if any of the cluster anchor keywords appears in the lexical field of the ground truth label, OR if there is no overlap with any of the fields and the ground truth label is OTHER.
Any other output is incorrect.

The accuracy is defined as correct\_outputs/total\_outputs. Using this metric, we achieved accuracies between 94-96%, depending on the initialization point of the clustering algorithm. Evaluation is automatically run when a new model is generated.

An interesting thing is that looking at the incorrect predictions, many of them can fall in two clusters, so the categorizations are not completely wrong.

For example, the question:
What do you think about the government support for electric car adoption? - Ground truth POLITICS/COUNTRY - Model output CARS, ELECTRIC, IMPORTANT, CAR, THINK
could fall in any of the POLITICS and ELECTRIC CARS clusters.


# Artifacts:
The following artifacts are generated and can be examined in the package:

Under /data:

- clustering\_results.txt - the dataset organized by clusters (including their thematic anchors). Questions in each cluster are sorted in decreasing order based on the centrality to the cluster centroid (1 - cosine similarity).

- incorrect\_predictions.txt - the dataset model outputs which did not match the LLM labels during evaluation.

Under /visualization:

- a 3D scatter plot of the clusters of embeddings, after 3D PCA was applied.

