from  dataset_loading import lazy_load_dataset
from preprocessing import clean
from preprocessing import meaningful
from preprocessing import corpus_properties
from preprocessing import save_positive_pairs
from preprocessing import load_positive_pairs
from network import NeuralNetwork
import analysis

import matplotlib.pyplot as plt
import numpy as np

# The first n lines of the dataset to be loaded.
n_lines = 100

# Compute immutable properties of the corpus and co
word2idx, idx2word, unigram = corpus_properties(meaningful(clean(lazy_load_dataset(n_lines = n_lines))))

# Compute skip-gram training pairs of (target_idx, context_idx) and persist to storage.
context_size = 2
save_positive_pairs(meaningful(clean(lazy_load_dataset(n_lines = n_lines))), word2idx, context_size)

# Hyperparameters during training
vec_size = len(word2idx)
neg_per_pos, n_epochs, learning_rate = 20, 2, 0.02

# Training loop: comment this entire block out to use saved word2vec
# Create a neural network that learns word2vec embeddings as one of its two weight matrices.
model = NeuralNetwork(word2idx, idx2word, unigram, vec_size)
model.train_epochs(load_positive_pairs, neg_per_pos, n_epochs, learning_rate)
np.savez(f"saves/trained_{n_lines}_{context_size}_{vec_size}_{neg_per_pos}_{n_epochs}_{learning_rate}_.npz", word2idx=word2idx, idx2word=idx2word, word2vec=model.word2vec)
plt.plot(model.model_wise_losses)
plt.show()


# model quality assess eval analysis
trained = np.load(f"saves/trained_{n_lines}_{context_size}_{vec_size}_{neg_per_pos}_{n_epochs}_{learning_rate}_.npz", allow_pickle=True)
analysis.analyse(trained)