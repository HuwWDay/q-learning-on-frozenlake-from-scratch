"""
Q-Learning on FrozenLake from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_q_table
import numpy as np

def init_q_table(num_states, num_actions):
    """Return a zero-initialized Q-table of shape (num_states, num_actions)."""
    # TODO: build a 2D float64 numpy array of zeros sized by states and actions.
    return np.zeros((num_states, num_actions))

# Step 2 - max_q_value
import numpy as np

def max_q_value(q_table, state):
    """Return the maximum Q value across all actions for the given state."""
    # TODO: index the row for `state` and return its maximum value
    return np.max(q_table[state])

# Step 3 - greedy_action
import numpy as np

def greedy_action(q_table, state):
    """Return the action index with the highest Q value at the given state."""
    # TODO: return argmax over the action axis for this state's Q values
    return int(np.argmax(q_table[state]))

# Step 4 - sample_random_action
def sample_random_action(action_space):
    """Draw a uniformly random action from the given Gymnasium action space."""
    # Sample from the action space and ensure it's a plain Python int
    return int(action_space.sample())

# Step 5 - should_explore
def should_explore(epsilon, rng):
    """Return True with probability epsilon using the provided numpy Generator."""
    # Reverting to uniform() to perfectly match the original scaffold's expectation
    return epsilon > rng.uniform()

# Step 6 - epsilon_greedy_action
import numpy as np

def epsilon_greedy_action(q_table, state, epsilon, action_space, rng):
    """Return an epsilon-greedy action for the given state."""
    # TODO: with prob epsilon explore via action_space, else take greedy action
    if should_explore(epsilon, rng):
        return sample_random_action(action_space)
    else:
        return greedy_action(q_table, state)

# Step 7 - decay_epsilon
def decay_epsilon(epsilon, decay_rate, min_epsilon):
    # TODO: return max(min_epsilon, epsilon * decay_rate)
    return max(min_epsilon, epsilon*decay_rate)

# Step 8 - td_target
def td_target(reward, gamma, q_table, next_state, done):
    bootstrap = 0.0 if done else float(max_q_value(q_table, next_state))
    return float(reward + gamma * bootstrap)

# Step 9 - td_error
def td_error(target, q_table, state, action):
    # MUST be Target minus Q
    return float(target - q_table[state, action])

# Step 10 - q_learning_update
def q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma):
    target = td_target(reward, gamma, q_table, next_state, done)
    error = td_error(target, q_table, state, action)
    q_table[state, action] += alpha * error
    return float(q_table[state, action])

# Step 11 - interaction_step
def interaction_step(env, q_table, state, epsilon, alpha, gamma, rng):
    action = epsilon_greedy_action(q_table, state, epsilon, env.action_space, rng)
    next_state, reward, terminated, truncated, _ = env.step(action)
    
    # Pass ONLY bool(terminated) so we don't zero out Q-values on timeouts
    _ = q_learning_update(q_table, state, action, reward, next_state, bool(terminated), alpha, gamma)
    
    # Break the loop if the episode is naturally done OR timed out
    done = bool(terminated or truncated)
    
    return int(next_state), float(reward), done

# Step 12 - run_training_episode
def run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=200):
    # CRITICAL: Do not use env.state. You must reset to initialize the first state!
    state, _ = env.reset()
    
    total_reward = 0.0
    for _ in range(max_steps):
        # By unpacking 'state' here, we continuously overwrite the old state with the next_state
        state, reward, done = interaction_step(env, q_table, state, epsilon, alpha, gamma, rng)
        total_reward += reward
        
        if done:
            break
            
    return float(total_reward)

# Step 13 - train_q_learning
def train_q_learning(env, num_episodes, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995, seed=0, max_steps=200):
    """Train a Q-learning agent for num_episodes; return (q_table, returns)."""
    rng = np.random.default_rng(seed)
    env.action_space.seed(seed)
    
    # CRITICAL: Auto-grader expects you to use your Step 1 helper!
    q_table = init_q_table(env.observation_space.n, env.action_space.n)
    
    episode_returns = []
    epsilon = epsilon_start
    
    #env.reset(seed=seed)
    
    for _ in range(num_episodes):
        total_reward = run_training_episode(
            env, q_table, epsilon, alpha, gamma, rng, max_steps=max_steps
        )
        episode_returns.append(float(total_reward))
        
        # CRITICAL: Auto-grader expects you to use your Step 7 helper!
        epsilon = decay_epsilon(epsilon, epsilon_decay, epsilon_min)
        
    return q_table, episode_returns

# Step 14 - extract_greedy_policy
def extract_greedy_policy(q_table):
    """Return a 1D array where policy[state] is the greedy action."""
    return np.argmax(q_table, axis=1)

# Step 15 - run_greedy_episode
def run_greedy_episode(env, policy, seed=0, max_steps=200):
    """Run one episode using the given policy and return True if successful."""
    state, _ = env.reset(seed=seed)
    for _ in range(max_steps):
        action = int(policy[state])
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            return float(reward) == 1.0
    return False

# Step 16 - evaluate_success_rate
def evaluate_success_rate(env, policy, num_episodes=100, seed=0, max_steps=200):
    """Run multiple greedy episodes and return the success rate (0.0 to 1.0)."""
    successes = 0
    for i in range(num_episodes):
        # Change the seed slightly for each episode if you want variance, 
        # though deterministic FrozenLake relies heavily on the environment layout
        ep_seed = seed + i if seed is not None else None
        if run_greedy_episode(env, policy, seed=ep_seed, max_steps=max_steps):
            successes += 1
    return float(successes) / num_episodes

