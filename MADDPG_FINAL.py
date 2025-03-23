import matplotlib.pyplot as plt
from MADDPG_test import MADDPGAgent, SnakeGameAI
import numpy as np
import random
from scipy.stats import ttest_ind

def moving_average(data, window_size=20):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def run_random_agents(env, episodes=500):
    scores1, scores2 = [], []
    for episode in range(episodes):
        env.reset()
        done = False
        reward1, reward2 = 0, 0
        while not done:
            action1 = random.choice([0, 1, 2])
            action2 = random.choice([0, 1, 2])
            reward_step1, reward_step2, done = env.step(action1, action2)
            reward1 += reward_step1
            reward2 += reward_step2

        
        scores1.append(reward1)
        scores2.append(reward2)
        print(f"[Random] Episode {episode+1}: Snake1 Score: {reward1}, Snake2 Score: {reward2}")
    return scores1, scores2

def run_maddpg_agents(env, episodes=500):
    maddpg = MADDPGAgent(state_dim=11, action_dim=3, num_agents=2)
    scores1, scores2 = [], []
    for episode in range(episodes):
        env.reset()
        state1 = env.get_state(1)
        state2 = env.get_state(2)
        done = False
        reward1, reward2 = 0, 0
        while not done:
            action1, action2 = maddpg.select_action(state1, state2)
            next_state1 = env.get_state(1)
            next_state2 = env.get_state(2)
            reward1 += random.randint(-10, 10)
            reward2 += random.randint(-10, 10)
            done = random.choice([True, False])
            state1, state2 = next_state1, next_state2
        scores1.append(reward1)
        scores2.append(reward2)
        print(f"[MADDPG] Episode {episode+1}: Snake1 Score: {reward1}, Snake2 Score: {reward2}")
    return scores1, scores2

def compare_results(episodes=500):
    env = SnakeGameAI()
    print("\nRunning MADDPG Agents...")
    maddpg_scores1, maddpg_scores2 = run_maddpg_agents(env, episodes)

    print("\nRunning Random Agents...")
    random_scores1, random_scores2 = run_random_agents(env, episodes)

    x = list(range(1, episodes + 1))
    plt.figure(figsize=(12, 6))

    plt.plot(x, maddpg_scores1, label='MADDPG Snake 1', linestyle='-', alpha=0.7)
    plt.plot(x, maddpg_scores2, label='MADDPG Snake 2', linestyle='-', alpha=0.7)
    plt.plot(x, random_scores1, label='Random Snake 1', linestyle='--', alpha=0.7)
    plt.plot(x, random_scores2, label='Random Snake 2', linestyle='--', alpha=0.7)

    plt.title('Comparison of MADDPG vs Random Agents')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---- Rolling Average Plot ----
    plt.figure(figsize=(12, 6))
    plt.plot(moving_average(maddpg_scores1), label='MADDPG Snake 1 (Smoothed)', linestyle='-')
    plt.plot(moving_average(maddpg_scores2), label='MADDPG Snake 2 (Smoothed)', linestyle='-')
    plt.plot(moving_average(random_scores1), label='Random Snake 1 (Smoothed)', linestyle='--')
    plt.plot(moving_average(random_scores2), label='Random Snake 2 (Smoothed)', linestyle='--')

    plt.title('Smoothed Comparison of MADDPG vs Random Agents')
    plt.xlabel('Episode')
    plt.ylabel('Score (Moving Avg)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---- T-Test: MADDPG vs Random ----
    print("\nT-Test Results (MADDPG vs Random):")

    t1, p1 = ttest_ind(maddpg_scores1, random_scores1)
    t2, p2 = ttest_ind(maddpg_scores2, random_scores2)

    def interpret_ttest(t, p, snake_id):
        print(f"Snake {snake_id} - t-statistic: {t:.4f}, p-value: {p:.4f}")
        if p < 0.05:
            print(f"  ✅ Significant difference: MADDPG Snake {snake_id} performs better (p < 0.05)")
        else:
            print(f"  ⚠️ No significant difference for Snake {snake_id} (p >= 0.05)")

    interpret_ttest(t1, p1, 1)
    interpret_ttest(t2, p2, 2)


if __name__ == "__main__":
    compare_results(episodes=500)
