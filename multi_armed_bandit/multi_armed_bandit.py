import numpy as np

def policy(Q=None, N=None, policy_name=""):
    random_dummy = np.zeros(100)
    random_dummy[3] = 1

    if policy_name == "Epsilon-greedy":
        if np.random.choice(random_dummy, 1) == 1: # probability 0.01
            A = np.random.randint(0, len(Q))
        else: # probability 0.99
            A = np.argmax(Q)

    elif policy_name == "UCB":
        c = 5 # for exploration
        if np.random.choice(random_dummy, 1) == 1: # probability 0.01
            A = np.random.randint(0, len(Q))
        else: # probability 0.99
            exploration = np.sqrt(np.log(i+1) / N)
            exploration[np.isnan(exploration)] = 0
            exploration[np.isinf(exploration)] = 0
            exploration *= c
            A = np.argmax(Q + exploration)

    return A

def bandit(A):
    mu = np.array([10, 4, 5, 2, 7, 8, 3, 9, 4, 1])
    sigma = np.array([0.3, 0.65, 0.7, 0.6, 0.4, 0.55, 0.45, 0.35, 0.4, 0.5])
    return np.random.normal(mu[A], sigma[A])

if __name__ == "__main__":
    n_arms = 10
    n_episodes = 1000

    Q = np.zeros(n_arms) # the estimated value of action
    # Q = np.array([10] * n_arms) # Optimistic initial
    N = np.zeros(n_arms) # the number of selected action

    for i in range(n_episodes):
        A = policy(Q, policy_name="Epsilon-greedy")
        # A = policy(Q, N, policy_name="UCB")
        
        R = bandit(A)
        N[A] += 1
        # update
        # step_size = 1 / N[A]
        step_size = 0.1
        Q[A] = Q[A] + step_size * (R - Q[A])

    print(Q)