# Q = XWq : What am I searching for?
# K = XWk : What information do I contain?
# V = XWv : What information do I pass forward?
#
# X is the input embedding
# Wk,Wq,Wv are the projection matrix

import numpy as np

def compute_attention_weights(emb_dim, embeddings):

    # Shape : (emb_dim,emb_dim) 
    # transforms one vector of length emb_dim into another vector of length emb_dim.
    Wq = np.random.randn(emb_dim, emb_dim)
    Wk = np.random.randn(emb_dim, emb_dim)
    Wv = np.random.randn(emb_dim, emb_dim)

    Q = embeddings @ Wq
    K = embeddings @ Wk
    V = embeddings @ Wv


    # Shape will be now (seq_len,emb_dim)
    print("Q shape:", Q.shape)
    print("K shape:", K.shape)
    print("V shape:", V.shape)

    # Compute attention score
    # compare every token query with every token key
    score = Q @ K.transpose()

    # Each row -> “How much does THIS token care about every other token?”
    # For 1st row E.g [7.19, -2.91, 3.00, 17.38, 10.61, ...]
    # Token 1 Strongly attends to Token 4 as 17.38 is very large
    print(f"Score : {score}")

    #Scaling

    # divide by sqrt(emb_dim) as dot product gets larger when the embedding dimension gets larger
    # We need to pass this weigths to softmax, and softmax is very sensitive to large numbers.
    scaled_scores = score / np.sqrt(emb_dim)

        # values become probabilities, each row sums to 1
    weights = softmax(scaled_scores)

    print(f"Weights = {weights}")
    
    return weights



def softmax(x):
    # Stable softmax trick :
    # subtract the largest value before exponentiation, without this trick Exponentials grow extremely fast
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
