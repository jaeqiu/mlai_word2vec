import numpy as np

BORING_WORDS = {
    "the",
    "a",
    "i",
    "and",
    "to",
    "it",
    "is",
    "thing",
    "after",
    "made",
    # , "pedals"
    "got",
    # , "guitar"
    "still",
    # , "sounds"
    # , "product"
    # , "tuner"
    "over",
    "way",
    "want",
    "into",
    "now",
    "been",
    "am",
    # , "playing"
    "buy",
    # , "work"
    # , "string"
    # , "mic"
    # , "picks"
    # , "guitars"
    "play",
    "even",
    "by",
    "stand",
    "need",
    # , "strap"
    "bought",
    "because",
    "using",
    "too",
    # , "easy"
    # , "tone"
    # , "time"
    "does",
    "any",
    # , "better"
    "im",
    # , "nice"
    "which",
    "there",
    # , "used"
    "also",
    # , "quality"
    "only",
    # , "works"
    "ive",
    "much",
    "had",
    # , "amp"
    # , "little"
    "do",
    "about",
    "dont",
    "what",
    "some",
    "other",
    # , "price"
    # , "really"
    "me",
    "no",
    "would",
    "has",
    "than",
    "from",
    # , "pedal"
    "get",
    "when",
    "up",
    "them",
    "will",
    "out",
    "an",
    "your",
    "well",
    "more",
    "at",
    "all",
    "can",
    # , "strings"
    # , "sound"
    "just",
    "use",
    # , "good"
    # , "very"
    # , "like"
    "was",
    "be",
    # , "great"
    "or",
    "if",
    "these",
    "one",
    "they",
    "so",
    "its",
    "as",
    "are",
    "not",
    "have",
    "but",
    "in",
    "you",
    "on",
    "with",
    "that",
    "my",
    "this",
    "for",
    "of",
}


def meaningful(gen):
    for sentence in gen:
        l = list(
            filter(
                lambda word: word not in BORING_WORDS,
                sentence,
            )
        )
        if l:
            yield (l)


def clean(sentence_generator):
    # keep lowercased alphas and spaces
    # tokenize to list of word strings
    for sentence in sentence_generator:
        tokenised_sentence = ""
        for char in sentence:
            if char.isalpha():
                tokenised_sentence += char.lower()
            if char.isspace() or char in {".", ","}:
                tokenised_sentence += " "
        yield (tokenised_sentence.split())


def corpus_properties(sentence_generator):

    word2idx: dict[str:int] = {}
    idx2occurences: dict[str:int] = {}

    for sentence in sentence_generator:
        for center_word in sentence:
            if center_word not in word2idx:
                idx = len(word2idx)
                idx2occurences[idx] = 1
                word2idx[center_word] = idx
            else:
                idx2occurences[word2idx[center_word]] += 1

    idx2word: dict[int:str] = {idx: word for word, idx in word2idx.items()}

    # for word in sorted(idx2word.values(), key = lambda x: len(x), reverse=True):
    #     print(word)

    unigram = np.array(list(idx2occurences.values())) ** (3 / 4)
    unigram /= np.sum(unigram)

    print()
    print(
        f"The vocabulary consists of {len(word2idx)} unique words for word2vec to learn the embedding of"
    )
    print(f"The most occurring words are:")
    for i, (k, v) in enumerate(
        sorted(idx2occurences.items(), key=lambda t: t[1], reverse=True)
    ):
        if i < 5:
            print(f"    {idx2word[k]} with {v} occurences")
    return word2idx, idx2word, unigram


def save_positive_pairs(
    sentence_generator,
    word2idx,
    context_size,
    save_file="saves/positive_training_pairs.txt",
):
    with open(save_file, "w") as f:
        print(f"Cleared {save_file}.")
        n = 0
        flat_samples = []
        for sentence in sentence_generator:
            for i, target in enumerate(sentence):
                if target in BORING_WORDS:
                    continue

                target_idx = word2idx[target]
                for j in range(i - context_size, i + context_size + 1):
                    if j < 0 or j >= len(sentence):
                        # Skip generation of samples if the context is out of sentence bounds
                        continue

                    if j == i:
                        # let's not dilute the effect of signals with self pairs
                        continue

                    positive = sentence[j]
                    if positive in BORING_WORDS:
                        continue
                    positive_idx = word2idx[positive]
                    flat_samples.append((target_idx, positive_idx))
                    n += 1
                    f.write(f"{target_idx} {positive_idx}\n")
        print(f"Wrote {n} training pairs to {save_file}.")


def load_positive_pairs(save_file="positive_training_pairs.txt"):
    with open(save_file, "r") as f:
        for line in f:
            target_idx, positive_idx = (int(x) for x in line.split())
            yield ((target_idx, positive_idx))
