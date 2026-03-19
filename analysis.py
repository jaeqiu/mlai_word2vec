import numpy as np

def analyse(trained_model_state):
    word2idx = trained_model_state["word2idx"].item()  # objects, not np arrays
    idx2word = trained_model_state["idx2word"].item()  # objects, not np arrays
    word2vec = trained_model_state["word2vec"]


    def top_k_similar(word2vec, idx2word, k):
        """Print top k nearest word pairs from trained word2vec."""
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
        print(f"\nTop {k} most similar word pairs")
        for a, b, s in pairs[:k*2:2]:
            print(f"    {a, b, s}")
    top_k_similar(word2vec, idx2word, k=12)

    def analogy(king, man, woman, k=4):
        for w in (king, man, woman):
            if w not in word2idx:
                print(f"'{w}' not in vocabulary")
                return

        normalized = word2vec / np.linalg.norm(word2vec, axis=1, keepdims=True)

        target = normalized[word2idx[king]] - normalized[word2idx[man]] + normalized[word2idx[woman]]

        similarities = normalized @ target
        for w in (king, man, woman):
            similarities[word2idx[w]] = 0
        
        sorted = np.argsort(similarities)[::-1]
        for i in range(k):
            result = idx2word[sorted[i]]
            print(f"\n{king} - {man} + {woman} ≈  {result} ({(similarities[sorted[i]]):.4f})")
    
    # analogy("king", "man", "woman")
    # analogy("paris", "france", "germany")
