import numpy as np


class NeuralNetwork:
    def __init__(self, word2idx, idx2word, unigram_distribution, vec_size):
        """Initialize model parameters and buffers based on the corpus properties and the vec size of the to be learned embeddings."""
        self.word2idx: dict[str:int] = word2idx
        self.idx2word: dict[int:str] = idx2word
        self.unigram_distribution: np.ndarray = unigram_distribution

        vocab_size = len(self.unigram_distribution)
        self.word2vec = np.random.default_rng().normal(size=(vocab_size, vec_size))
        self.vec2context = np.random.default_rng().normal(size=(vocab_size, vec_size))
        self.word2vec_gradients = np.zeros(self.word2vec.shape)
        self.vec2context_gradients = np.zeros(self.vec2context.shape)
        self.model_wise_losses = []
        print(
            f"\nInitialised a neural network that learns the {vec_size}-dimensional embeddings of {len(word2idx)} unique words."
        )

    def forward(self, batch):
        """Batch forward pass (predictions, losses, gradients)"""
        targets, contexts, grounds = batch.T

        tar_vecs = self.word2vec[targets]
        con_vecs = self.vec2context[contexts]

        preactivations = np.sum(tar_vecs * con_vecs, axis=1)
        preactivations = np.clip(preactivations, -100, 100)
        predictions = 1 / (1 + np.exp(-preactivations))

        # Binary Cross Entropy Loss
        def safe_log(x):
            return np.log(np.clip(x, 1e-15, 1 - 1e-15))

        losses = np.where(
            grounds == 1, -safe_log(predictions), -safe_log(1 - predictions)
        )

        # gradient of loss w.r.t. preactivation vectors(pairwise dots)
        upstreams = np.where(grounds == 1, predictions - 1, predictions - 0)

        self.word2vec_gradients[targets] += upstreams[:, None] * con_vecs
        self.vec2context_gradients[contexts] += upstreams[:, None] * tar_vecs

        return losses

    def apply_gradient(self, learning_rate):
        """Hyperparameter update: apply the accumulated gradients at a learning rate sized step."""
        self.word2vec -= self.word2vec_gradients * learning_rate
        self.vec2context -= self.vec2context_gradients * learning_rate

        self.word2vec_gradients = np.zeros(self.word2vec.shape)
        self.vec2context_gradients = np.zeros(self.vec2context.shape)

    def train(self, tuple, neg_per_pos, learning_rate):
        """Negative sampling: Take a single positive pair and create n negative pairs. Train on the batch of 1+n pairs.
        """
        target, positive = tuple

        batch = []
        batch.append((target, positive, 1))
        for _ in range(neg_per_pos):
            negative = None
            while negative is None or negative == target or negative == positive:
                negative = np.random.choice(
                    a=list(self.idx2word.keys()),
                    size=None,
                    replace=True,
                    p=self.unigram_distribution,
                )
            batch.append((target, negative, 0))
        batch = np.array(batch)

        batch_loss = self.forward(batch)
        self.apply_gradient(learning_rate)
        return batch_loss

    def train_epochs(self, center_sample_pairs, neg_per_pos, n_epochs, learning_rate, patience = 10):
        """Train for multiple epochs over sampled pairs."""
        print(f"\nStart training")

        epoch_losses = []
        best_epoch_loss = np.inf
        count = patience
        for epoch in range(n_epochs):

            sample_losses = []

            # for center_sample_pair in center_sample_pairs:
            for k, center_sample_pair in enumerate(center_sample_pairs()):
                # print(k)
                sample_losses.append(
                    self.train(center_sample_pair, neg_per_pos, learning_rate)
                )
            mean_batchloss = np.mean(sample_losses)
            epoch_losses.append(mean_batchloss)

            print(
                f"Finished epoch {epoch}, model scored a mean batch-loss of{epoch_losses[-1]}"
            )

            if (mean_batchloss < best_epoch_loss):
                best_epoch_loss = np.mean(sample_losses)
                count = patience
            else:
                count -= 1
                if (count <= 0):
                    print(f"Early stopping since loss hasn't beat {best_epoch_loss} in {patience} epochs")
                    break

        self.model_wise_losses.extend(epoch_losses)

        print(f"Finished training")

        return epoch_losses
