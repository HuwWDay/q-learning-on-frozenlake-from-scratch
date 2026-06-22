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
    # TODO: draw a uniform sample from rng and compare it to epsilon
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
import numpy as np

def td_target(reward, gamma, q_table, next_state, done):
    """Compute r + gamma * max_a Q(next_state, a), zeroing the bootstrap when done."""
    # If the episode is done, the bootstrap value is 0. Otherwise, it's max_a Q(s', a)
    bootstrap = 0.0 if done else max_q_value(q_table, next_state)
    
    # Return the complete TD target as a plain float
    return float(reward + gamma * bootstrap)

# Step 9 - td_error
def td_error(target, q_table, state, action):
    # TODO: return the TD error: target minus current Q(state, action)
    return target - q_table[state][action]

# Step 10 - q_learning_update
def q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma):
    # TODO: apply Q(s,a) += alpha * (target - Q(s,a)) in place and return the new Q value
    target = td_target(reward, gamma, q_table, next_state, done)
    error = td_error(target, q_table, state, action)
    q_table[state][action] += alpha*error
    return float(q_table[state, action])

# Step 11 - interaction_step
def interaction_step(env, q_table, state, epsilon, alpha, gamma, rng):
    """Select an action, step the env, update the Q-table, and return the step results."""
    # 1. Select an action using the epsilon-greedy policy helper
    action = epsilon_greedy_action(q_table, state, epsilon, env.action_space, rng)
    
    # 2. Step the environment using the chosen action
    # Gymnasium returns: next_state, reward, terminated, truncated, info
    next_state, reward, terminated, truncated, _ = env.step(action)
    
    # 3. Combine terminated and truncated to determine if the episode is done
    done = bool(terminated or truncated)
    
    # 4. Apply the Q-learning update in-place
    _ = q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma)
    
    # 5. Return the tuple with strict plain Python types
    return int(next_state), float(reward), done

# Step 12 - run_training_episode
def run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=200):
    # This resets the frozen lake back to the starting tile 'S' for the new episode!
    state, _ = env.reset() 
    
    total_reward = 0.0
    step = 0
    
    while step < max_steps:
        state, reward, done = interaction_step(env, q_table, state, epsilon, alpha, gamma, rng)
        total_reward += reward
        step += 1
        if done:
            break
            
    return float(total_reward)

# Step 13 - train_q_learning
import numpy as np

def train_q_learning(env, num_episodes, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995, seed=0, max_steps=200):
    """Train a Q-learning agent for num_episodes; return (q_table, returns)."""
    rng = np.random.default_rng(seed)
    env.action_space.seed(seed)
    
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))
    
    episode_returns = []
    epsilon = epsilon_start
    
    # We pass the base seed to the environment's initial state once to lock the map's layout generation trajectory
    env.reset(seed=seed)
    
    for episode in range(num_episodes):
        # NOTE: run_training_episode must handle calling env.reset() EVERY time it starts.
        total_reward = run_training_episode(
            env, q_table, epsilon, alpha, gamma, rng, max_steps=max_steps
        )
        
        episode_returns.append(float(total_reward))
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
    return q_table, episode_returns

# Step 14 - extract_greedy_policy (not yet solved)
# TODO: implement

# Step 15 - run_greedy_episode (not yet solved)
# TODO: implement

# Step 16 - evaluate_success_rate (not yet solved)
# TODO: implement

