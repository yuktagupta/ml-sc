import numpy as np 
import random 
# Define the environment 
grid_size = 3  # Smaller grid 
goal_state = (2, 2) 
obstacle_state = (1, 1)  # Single obstacle 
actions = ['up', 'down', 'left', 'right'] 
action_to_delta = { 
    'up': (-1, 0), 
    'down': (1, 0), 
    'left': (0, -1), 
    'right': (0, 1) 
} 
# Initialize Q-table (simple 3D array for states and actions) 
q_table = np.zeros((grid_size, grid_size, len(actions))) 
# Parameters 
alpha = 0.1  # Learning rate 
gamma = 0.9  # Discount factor 
epsilon = 1.0  # Exploration rate 
epsilon_decay = 0.99 
min_epsilon = 0.1 
episodes = 200  # Fewer episodes 
# Reward function 
def get_reward(state): 
    if state == goal_state: 
        return 10  # Reward for reaching the goal 
    elif state == obstacle_state: 
        return -10  # Penalty for hitting the obstacle 
    return -1  # Step penalty 
# Check if the new state is valid 
def is_valid_state(state): 
    return 0 <= state[0] < grid_size and 0 <= state[1] < grid_size and state != obstacle_state 
# Main Q-learning loop 
for episode in range(episodes): 
    state = (0, 0)  # Start at the top-left corner 
    total_reward = 0 
    while state != goal_state: 
        # Choose an action (epsilon-greedy strategy) 
        if random.uniform(0, 1) < epsilon: 
            action = random.choice(actions)  # Explore 
        else: 
            action = actions[np.argmax(q_table[state[0], state[1]])]  # Exploit best action 
        # Perform the action 
        delta = action_to_delta[action] 
        next_state = (state[0] + delta[0], state[1] + delta[1]) 
        # Stay in the same state if the move is invalid 
        if not is_valid_state(next_state): 
            next_state = state 
        # Get reward and update Q-table 
        reward = get_reward(next_state) 
        total_reward += reward 
        best_next_action = np.max(q_table[next_state[0], next_state[1]]) 
        q_table[state[0], state[1], actions.index(action)] += alpha * ( 
            reward + gamma * best_next_action - q_table[state[0], state[1], actions.index(action)] 
        ) 
        # Update state 
        state = next_state 
    # Decay epsilon 
    epsilon = max(min_epsilon, epsilon * epsilon_decay) 
print(f"Episode {episode + 1}: Total Reward = {total_reward}") 
# Display the learned policy 
policy = np.full((grid_size, grid_size), ' ') 
for i in range(grid_size): 
    for j in range(grid_size): 
        if (i, j) == goal_state: 
            policy[i, j] = 'G'  # Goal 
        elif (i, j) == obstacle_state: 
            policy[i, j] = 'X'  # Obstacle 
        else: 
            best_action = np.argmax(q_table[i, j]) 
            policy[i, j] = actions[best_action][0].upper()  # First letter of the best action 
print("Learned Policy:") 
print(policy) 
