
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from loguru import logger
import random



class Bandit(ABC):
    """ """
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        """ """
        pass

    @abstractmethod
    def update(self):
        """ """
        pass

    @abstractmethod
    def experiment(self):
        """ """
        pass

    @abstractmethod
    def report(self):
        """ """
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """ """
    def __init__(self, n_bandits, epsilon=1.0):
        # EpsilonGreedy-specific attributes
        self.epsilon = epsilon
        self.q_values = [0] * n_bandits  # Estimated rewards for each bandit
        self.bandit_counts = [0] * n_bandits  # Number of times each bandit is pulled
        self.n_bandits = n_bandits  # Total number of bandits

    def pull(self):
        """ """
        if random.random() < self.epsilon:
            return np.random.choice(self.n_bandits)  # Explore
        return np.argmax(self.q_values)  # Exploit

    def update(self, bandit_idx, reward):
        """

        :param bandit_idx: 
        :param reward: 

        """
        self.bandit_counts[bandit_idx] += 1
        # Update the Q-value for the chosen bandit
        self.q_values[bandit_idx] += (reward - self.q_values[bandit_idx]) / self.bandit_counts[bandit_idx]

    def experiment(self, n_trials, bandit_probabilities):
        """

        :param n_trials: 
        :param bandit_probabilities: 

        """
        rewards = []
        chosen_bandits = []
        self.epsilon = 1.0
        for t in range(1, n_trials + 1):
            self.epsilon = 1 / t  # Decay epsilon
            bandit_idx = self.pull()
            reward = np.random.binomial(1, bandit_probabilities[bandit_idx])  # Simulate reward
            rewards.append(reward)
            chosen_bandits.append(bandit_idx)
            self.update(bandit_idx, reward)
        return rewards, chosen_bandits

    def report(self, rewards, chosen_bandits, optimal_reward, csv_file="epsilon_greedy_results.csv"):
        """

        :param rewards: 
        :param chosen_bandits: 
        :param optimal_reward: 
        :param csv_file:  (Default value = "epsilon_greedy_results.csv")

        """
        data = pd.DataFrame({
            "Bandit": chosen_bandits,
            "Reward": rewards,
            "Algorithm": ["Epsilon-Greedy"] * len(rewards)
        })
        data.to_csv(csv_file, index=False)

        cumulative_rewards = np.cumsum(rewards)
        cumulative_regret = np.cumsum(optimal_reward - np.array(rewards))

        logger.info(f"Cumulative Reward (Epsilon-Greedy): {cumulative_rewards[-1]}")
        logger.info(f"Cumulative Regret (Epsilon-Greedy): {cumulative_regret[-1]}")

        return cumulative_rewards, cumulative_regret

    def __repr__(self):
        return (f"EpsilonGreedy(n_bandits={self.n_bandits}, "
                f"epsilon={self.epsilon}, "
                f"q_values={self.q_values}, "
                f"bandit_counts={self.bandit_counts})")


class ThompsonSampling(Bandit):
    """ """
    def __init__(self, n_bandits):
        # ThompsonSampling-specific attributes
        self.alpha = [1] * n_bandits  # Success counts for each bandit
        self.beta = [1] * n_bandits  # Failure counts for each bandit
        self.n_bandits = n_bandits  # Total number of bandits

    def pull(self):
        """ """
        # Sample from Beta distribution for each bandit
        sampled_values = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n_bandits)]
        return np.argmax(sampled_values)  # Choose the bandit with the highest sampled value

    def update(self, bandit_idx, reward):
        """

        :param bandit_idx: 
        :param reward: 

        """
        # Update the Beta distribution parameters for the chosen bandit
        if reward == 1:
            self.alpha[bandit_idx] += 1
        else:
            self.beta[bandit_idx] += 1

    def experiment(self, n_trials, bandit_probabilities):
        """

        :param n_trials: 
        :param bandit_probabilities: 

        """
        rewards = []
        chosen_bandits = []
        for _ in range(n_trials):
            bandit_idx = self.pull()
            reward = np.random.binomial(1, bandit_probabilities[bandit_idx])  # Simulate reward
            rewards.append(reward)
            chosen_bandits.append(bandit_idx)
            self.update(bandit_idx, reward)
        return rewards, chosen_bandits

    def report(self, rewards, chosen_bandits, optimal_reward, csv_file="thompson_sampling_results.csv"):
        """

        :param rewards: 
        :param chosen_bandits: 
        :param optimal_reward: 
        :param csv_file:  (Default value = "thompson_sampling_results.csv")

        """
        data = pd.DataFrame({
            "Bandit": chosen_bandits,
            "Reward": rewards,
            "Algorithm": ["Thompson Sampling"] * len(rewards)
        })
        data.to_csv(csv_file, index=False)

        cumulative_rewards = np.cumsum(rewards)
        cumulative_regret = np.cumsum(optimal_reward - np.array(rewards))

        logger.info(f"Cumulative Reward (Thompson Sampling): {cumulative_rewards[-1]}")
        logger.info(f"Cumulative Regret (Thompson Sampling): {cumulative_regret[-1]}")

        return cumulative_rewards, cumulative_regret

    def __repr__(self):
        return (f"ThompsonSampling(n_bandits={self.n_bandits}, "
                f"alpha={self.alpha}, "
                f"beta={self.beta})")


class Visualization():
    """ """

    def plot1(self, rewards, title="Epsilon-Greedy Learning Process"):
        """

        :param rewards: 
        :param title:  (Default value = "Epsilon-Greedy Learning Process")

        """
        # Visualize the performance of each bandit: linear and log
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(rewards), label="Cumulative Reward")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()


    def plot2(self, eg_rewards, eg_regret, ts_rewards, ts_regret):
        """

        :param eg_rewards: 
        :param eg_regret: 
        :param ts_rewards: 
        :param ts_regret: 

        """
        plt.figure(figsize=(12, 8))

        # Cumulative rewards
        plt.subplot(2, 1, 1)
        plt.plot(np.cumsum(eg_rewards), label="Epsilon-Greedy")
        plt.plot(np.cumsum(ts_rewards), label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Rewards Comparison")
        plt.legend()
        plt.grid()

        # Cumulative regret
        plt.subplot(2, 1, 2)
        plt.plot(eg_regret, label="Epsilon-Greedy Regret")
        plt.plot(ts_regret, label="Thompson Sampling Regret")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Regret")
        plt.title("Cumulative Regrets Comparison")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()


def comparison(eg_rewards, eg_regret, ts_rewards, ts_regret, eg_chosen_bandits, ts_chosen_bandits, n_bandits):
    """

    :param eg_rewards: 
    :param eg_regret: 
    :param ts_rewards: 
    :param ts_regret: 
    :param eg_chosen_bandits: 
    :param ts_chosen_bandits: 
    :param n_bandits: 

    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(15, 8))

    plt.subplot(2, 2, 1)
    plt.plot(np.cumsum(eg_rewards), label="Epsilon-Greedy")
    plt.plot(np.cumsum(ts_rewards), label="Thompson Sampling")
    plt.title("Cumulative Rewards")
    plt.xlabel("Trials")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(eg_regret, label="Epsilon-Greedy")
    plt.plot(ts_regret, label="Thompson Sampling")
    plt.title("Cumulative Regret")
    plt.xlabel("Trials")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.grid()

    eg_bandit_selection = [eg_chosen_bandits.count(b) for b in range(n_bandits)]
    ts_bandit_selection = [ts_chosen_bandits.count(b) for b in range(n_bandits)]

    plt.subplot(2, 2, 3)
    plt.bar(range(n_bandits), eg_bandit_selection, alpha=0.7, label="Epsilon-Greedy")
    plt.title("Bandit Selection Frequency (Epsilon-Greedy)")
    plt.xlabel("Bandit")
    plt.ylabel("Frequency")
    plt.xticks(range(n_bandits))
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.bar(range(n_bandits), ts_bandit_selection, alpha=0.7, label="Thompson Sampling", color="orange")
    plt.title("Bandit Selection Frequency (Thompson Sampling)")
    plt.xlabel("Bandit")
    plt.ylabel("Frequency")
    plt.xticks(range(n_bandits))
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


