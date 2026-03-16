import numpy as np
def analyse(trained_model_state):
    word2idx = trained_model_state["word2idx"].item()  # objects, not np arrays
    idx2word = trained_model_state["idx2word"].item()  # objects, not np arrays
    word2vec = trained_model_state["word2vec"]

    def top_k_similar(word2vec, idx2word, k):
        magnitudes = np.linalg.norm(word2vec, axis=1, keepdims=True)
        normalized_word2vec = word2vec / magnitudes

        pairs = []
        vocab = normalized_word2vec.shape[0]

        for i in range(vocab):
            similar = normalized_word2vec @ normalized_word2vec[i]
            # self similarity is trivial/redundant/useless
            similar[i] = -np.inf
            j = np.argmax(similar)
            pairs.append((idx2word[i], idx2word[j], similar[j]))

        # Descending similarity
        pairs.sort(key=lambda x: x[2], reverse=True)
        print()
        print(f"Top {k} most similar word pairs")
        for a, b, s in pairs[:k:2]:
            print(f"    {a, b, s}")
    top_k_similar(word2vec, idx2word, k=12)