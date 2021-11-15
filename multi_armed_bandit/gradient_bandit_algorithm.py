# softmax algorithm
import numpy as np

def bandit(A):
    mu = np.array([10, 4, 5, 2, 7, 8, 3, 9, 4, 1])
    sigma = np.array([0.3, 0.65, 0.7, 0.6, 0.4, 0.55, 0.45, 0.35, 0.4, 0.5])
    return np.random.normal(mu[A], sigma[A])

if __name__ == "__main__":
    n_arms = 10
    n_episodes = 1000
    H = np.zeros(n_arms) # preference
    mean_reward = 0
    lr = 0.1
    actions = np.arange(n_arms)

    N = np.zeros(n_arms) # the number of selected action

    for i in range(n_episodes):
        action_prob = np.exp(H) / np.sum(np.exp(H)) # probability of actions

        action = np.random.choice(actions, 1, p=action_prob)[0]
        # print(action)
        N[action] += 1

        R = bandit(action)
        advantage = R - mean_reward

        # update selected preference
        H[action] = H[action] + lr * advantage * (1 - action_prob[action])
        # update unselected preference
        unselected_action = np.setdiff1d(actions, action)
        H[unselected_action] = H[unselected_action] - lr * advantage * action_prob[unselected_action]
    print("N: ", N)
    print("H: ", H)