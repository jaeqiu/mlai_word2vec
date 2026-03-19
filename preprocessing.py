import numpy as np

def clean(sentence_generator):
    """Lowercase, keep alphabetic characters, and tokenize sentences."""
    for sentence in sentence_generator:
        tokenised_sentence = ""
        for char in sentence:
            if char.isalpha():
                tokenised_sentence += char.lower()
            if char.isspace() or char in {".", ","}:
                tokenised_sentence += " "
        yield (tokenised_sentence.split())

def subsample(sentence_generator, word2idx, idx2freq, t = 1e-3):
    for sentence in sentence_generator:
        result = []
        for token in sentence:
            unigram = np.array(list(idx2freq.values()), dtype=float)
            unigram /= np.sum(unigram)

            # Skip word with discard probability. Frequent words get discarded more often
            discard_probability = 1.0 - np.sqrt(t/unigram[word2idx[token]])
            if (np.random.default_rng().uniform() <= discard_probability):
                continue
            result.append(token)
        yield result

def corpus_properties(sentence_generator):
    """Build vocabulary and unigram distribution for negative sampling."""

    word2idx: dict[str:int] = {}
    idx2freq: dict[str:int] = {}

    for sentence in sentence_generator:
        for center_word in sentence:
            if center_word not in word2idx:
                idx = len(word2idx)
                idx2freq[idx] = 1
                word2idx[center_word] = idx
            else:
                idx2freq[word2idx[center_word]] += 1

    idx2word: dict[int:str] = {idx: word for word, idx in word2idx.items()}


    print(
        f"\nThe vocabulary consists of {len(word2idx)} unique words for word2vec to learn the embedding of"
    )
    print(f"The most occurring words are:")
    for i, (k, v) in enumerate(
        sorted(idx2freq.items(), key=lambda t: t[1], reverse=True)
    ):
        if i < 5:
            print(f"    {idx2word[k]} with {v} occurences")
    return word2idx, idx2word, idx2freq


def save_positive_pairs(
    sentence_generator,
    word2idx,
    context_size,
    save_file="saves/temp/positive_training_pairs.txt",
):
    """Write positive skip-gram pairs to storage."""
    with open(save_file, "w") as f:
        print(f"\nCleared {save_file}.")
        n = 0
        flat_samples = []
        for sentence in sentence_generator:
            for i, target in enumerate(sentence):
                target_idx = word2idx[target]
                for j in range(i - context_size, i + context_size + 1):
                    if j < 0 or j >= len(sentence):
                        # Skip generation of samples if the context is out of sentence bounds
                        continue

                    if j == i:
                        # Let's not dilute the effect of signals with self pairs
                        continue

                    positive = sentence[j]
                    positive_idx = word2idx[positive]
                    flat_samples.append((target_idx, positive_idx))
                    n += 1
                    f.write(f"{target_idx} {positive_idx}\n")
        print(f"Wrote {n} training pairs to {save_file}.")


def load_positive_pairs(save_file="saves/temp/positive_training_pairs.txt"):
    """Lazy load and parse positive pairs"""
    with open(save_file, "r") as f:
        for line in f:
            target_idx, positive_idx = (int(x) for x in line.split())
            yield ((target_idx, positive_idx))
