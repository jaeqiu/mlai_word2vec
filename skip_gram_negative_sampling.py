import numpy as np

class NeuralNetwork:
    def __init__(self, word2idx, idx2word, unigram_distribution, vec_size):
        self.word2idx: dict[str:int] = word2idx
        self.idx2word: dict[int:str] = idx2word
        self.unigram_distribution: np.ndarray = unigram_distribution

        vocab_size = len(self.unigram_distribution)
        self.word2vec = np.random.default_rng().normal(size=(vocab_size, vec_size))
        self.vec2context = np.random.default_rng().normal(size=(vocab_size, vec_size))
        self.word2vec_gradients = np.zeros(self.word2vec.shape)
        self.vec2context_gradients = np.zeros(self.vec2context.shape)
        self.model_wise_losses = []
        print()
        print(f"Initialised a neural network that learns the {vec_size}-dimensional embeddings of {len(word2idx)} unique words.")
    
    def forward(self, batch):
        targets, contexts, grounds = batch.T

        # target word embedding
        # context embedding for the singular context word

        tar_vecs = self.word2vec[targets]
        con_vecs = self.vec2context[contexts]
        

        preactivations = np.sum(tar_vecs * con_vecs, axis=1)
        # preactivations = np.dot(tar_vecs, con_vecs.T)
        preactivations = np.clip(preactivations, -100, 100)
        predictions = 1/(1+np.exp(-preactivations))

        # Binary Cross Entropy Loss
        #safe log loss by ensuring no division by zero, with epsilon
        # 1 - grounds will be 0 for pos, 1 for neg
        def safe_log(x):
            return np.log(np.clip(x, 1e-15, 1-1e-15))
        
        # if grounds == 1:
        #     loss = -safe_log(predictions)
        # else:
        #     loss = -safe_log(1 - predictions)

        losses = np.where(grounds == 1, -safe_log(predictions), -safe_log(1 - predictions))

        # loss = (1 - ground) * -safe_log(1 - prediction)  # case if negative
        # loss += (ground) * -safe_log(prediction)  # case if positive
        
        # In the -log(x) function, zero values inflate.
        # By orienting misclass at 0 and correct at 1,
        # we can use this such that false confidence is punished with a high loss.

        # gradient of loss w.r.t. preactivation vectors(pairwise dots)
        upstreams = np.where(grounds == 1, predictions - 1, predictions - 0)

        # if grounds == 1:
        #     gradient_loss_preactivation = predictions - 1
        # else:
        #     gradient_loss_preactivation = predictions - 0
        # gradient_loss_preactivation = prediction - ground for pos/negative
        self.word2vec_gradients[targets] += upstreams[:, None] * con_vecs
        self.vec2context_gradients[contexts] += upstreams[:, None] * tar_vecs
        # for i, target in enumerate(targets):
        #     self.word2vec_gradients[target] += (upstreams[i] * con_vecs[i])
        # for i, context in enumerate(contexts):
        #     self.vec2context_gradients[context] +=(upstreams[i] * tar_vecs[i])

        return losses


    def apply_gradient(self, learning_rate):
        self.word2vec -= self.word2vec_gradients * learning_rate
        self.vec2context -= self.vec2context_gradients * learning_rate

        self.word2vec_gradients = np.zeros(self.word2vec.shape)
        self.vec2context_gradients = np.zeros(self.vec2context.shape)

    def train(self, tuple, neg_per_pos, learning_rate):
        target, positive = tuple

        batch = []
        batch.append((target, positive, 1))
        for _ in range(neg_per_pos):
            negative = positive
            while (negative == positive):
                negative = np.random.choice(a=list(self.idx2word.keys()), size=None, replace=True, p=self.unigram_distribution)
            batch.append((target, negative, 0))
        batch = np.array(batch)

        batch_loss = self.forward(batch)
        self.apply_gradient(learning_rate)
        return batch_loss

    def train_epochs(self, center_sample_pairs, neg_per_pos, n_epochs, learning_rate):
        print(f"Start training")

        epoch_losses = []
        for i in range(n_epochs):
            sample_losses = []
            # for center_sample_pair in center_sample_pairs:
            for k, center_sample_pair in enumerate(center_sample_pairs()):
                
                # print(k)
                sample_losses.append(self.train(center_sample_pair, neg_per_pos, learning_rate))
            epoch_losses.append(np.mean(sample_losses))
            print(f"Finished epoch {i}, model scored a mean batch-loss of{epoch_losses[-1]}")
        self.model_wise_losses.extend(epoch_losses)

        print(f"Finished training")

        return epoch_losses
