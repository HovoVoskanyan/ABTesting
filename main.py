from Bandit import Bandit, ThompsonSampling, EpsilonGreedy, Visualization,comparison
from loguru import  logger



if __name__ == '__main__':
    # Define the true success probabilities of the bandits
    bandit_probabilities = [0.1, 0.3, 0.5, 0.7]

    # Number of trials
    n_trials = 20000

    # Run Epsilon-Greedy
    epsilon_greedy = EpsilonGreedy(n_bandits=len(bandit_probabilities))
    eg_rewards, eg_chosen_bandits = epsilon_greedy.experiment(n_trials, bandit_probabilities)

    # Run Thompson Sampling
    thompson_sampling = ThompsonSampling(n_bandits=len(bandit_probabilities))
    ts_rewards, ts_chosen_bandits = thompson_sampling.experiment(n_trials, bandit_probabilities)

    # Calculate the optimal reward (always choosing the best bandit)
    optimal_reward = max(bandit_probabilities)

    logger.info(f"Best Bandit Reward: {optimal_reward}")


    # Generate reports for Epsilon-Greedy
    eg_cumulative_rewards, eg_cumulative_regret = epsilon_greedy.report(
        rewards=eg_rewards,
        chosen_bandits=eg_chosen_bandits,
        optimal_reward=optimal_reward,
        csv_file="epsilon_greedy_results.csv"
    )

    # Generate reports for Thompson Sampling
    ts_cumulative_rewards, ts_cumulative_regret = thompson_sampling.report(
        rewards=ts_rewards,
        chosen_bandits=ts_chosen_bandits,
        optimal_reward=optimal_reward,
        csv_file="thompson_sampling_results.csv"
    )

    # Visualize the results
    visualization = Visualization()

    # Learning process visualizations
    visualization.plot1(eg_rewards, title="Epsilon-Greedy Learning Process")
    visualization.plot1(ts_rewards, title="Thompson Sampling Learning Process")

    # Compare cumulative rewards and regrets
    visualization.plot2(
        eg_rewards=eg_rewards,
        eg_regret=eg_cumulative_regret,
        ts_rewards=ts_rewards,
        ts_regret=ts_cumulative_regret
    )

    comparison(
        eg_rewards=eg_rewards,
        eg_regret=eg_cumulative_regret,
        ts_rewards=ts_rewards,
        ts_regret=ts_cumulative_regret,
        eg_chosen_bandits=eg_chosen_bandits,
        ts_chosen_bandits=ts_chosen_bandits,
        n_bandits=len(bandit_probabilities)
    )
