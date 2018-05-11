from __future__ import print_function
import numpy as np
import multiprocessing as mp


np.random.seed(0)


def worker_process(arg):
    get_reward_func, weights = arg
    return get_reward_func(weights)


class EvolutionStrategy(object):
    def __init__(self, weights, get_reward_func, population_size=50, sigma=0.1, learning_rate=0.03, decay=0.998,
                 num_threads=1):

        self.weights = weights
        self.get_reward = get_reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate
        self.decay = decay
        self.num_threads = mp.cpu_count() if num_threads == -1 else num_threads

    def _get_weights_try(self, w, p):
        weights_try = []
        for index, i in enumerate(p):
            jittered = self.SIGMA * i
            weights_try.append(w[index] + jittered)
        return weights_try

    def get_weights(self):
        return self.weights

    def run(self, iterations, print_step=10):
        pool = mp.Pool(self.num_threads) if self.num_threads > 1 else None
        for iteration in range(iterations):

            population = []
            for i in range(self.POPULATION_SIZE):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)

            if pool is not None:
                rewards = pool.map(worker_process,
                                   ((self.get_reward, self._get_weights_try(self.weights, individual)) for individual in
                                    population))
            else:
                rewards = []
                for individual in population:
                    weights_try = self._get_weights_try(self.weights, individual)
                    rewards.append(self.get_reward(weights_try))
            rewards = np.array(rewards)

            rewards = (rewards - np.mean(rewards)) / np.std(rewards)

            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = w + self.LEARNING_RATE / (self.POPULATION_SIZE * self.SIGMA) * np.dot(A.T,
                                                                                                            rewards).T

            self.LEARNING_RATE *= self.decay

            if (iteration + 1) % print_step == 0:
                print('iter %d. reward: %f' % (iteration + 1, self.get_reward(self.weights)))
        if pool is not None:
            pool.close()
            pool.join()
