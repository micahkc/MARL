from tkinter import Tk
from random import randint
import torch
import numpy as np
import matplotlib.pyplot as plt
# Import from other python modules.
import gui
from model import Agent
import pickle
def main():
    # Set learning parameters.
    num_episodes = 1000
    gamma = 0.999 # Discount factor.

    #env = gui.create_random_env2(num_drones=7, num_targets=5, num_obstacles=0)
    env = gui.create_default_env3()
    
    root = Tk()
    map = gui.Map(root,env)
    map.visual = False
    if not map.visual:
        root.destroy()

    # For each drone in the environment, create a learning agent (includes both policy and value nns).
    agents = {}
    for drone_id in range(1, env.num_drones+1):
        agents[drone_id] = Agent(drone_id)
        agents[drone_id].create_models()
    
    episode_rewards = []
    # Training loop
    for i in range(num_episodes):
        # Create empty history for observations, actions, and rewards for this episoide.
        done = False
        observations_history = {x:[] for x in range(1,env.num_drones+1)}
        actions_history = {x:[] for x in range(1,env.num_drones+1)}
        rewards_history = {x:[] for x in range(1,env.num_drones+1)}

        # Initial Observations
        observations = env.reset()
        
        # Run entire episoide.
        while not done:
            # Get actions from each drone's actor policy for this step.
            actions = {x:[] for x in range(1,env.num_drones+1)}
            for drone_id in range(1,env.num_drones+1):
                observation = torch.tensor(observations[drone_id], dtype=torch.float)
                action_dists = agents[drone_id].actor.forward(observation)
                actions_sampled = action_dists.sample().detach().data.numpy()
                actions[drone_id] = [actions_sampled[0], actions_sampled[1]]
                
                
            # Implement these actions in the env.
            next_observations, rewards, done = env.step(actions)
            
            # Update map visualization for this step if required.
            if map.visual:
                map.update_map(env)
            
            # Add actions, rewards and observations for this step to history.
            # Example actions_history = {1: [[0.3,0.4], [0.3, 0.5]], 2: [[0.2,0.4]]}
            for drone_id in range(1,env.num_drones+1):
                actions_history[drone_id].append(actions[drone_id])
                rewards_history[drone_id].append(rewards[drone_id])
                observations_history[drone_id].append(next_observations[drone_id])

        # Calculate cumulative rewards
        cum_rewards = {x:np.zeros_like(rewards_history[x]) for x in range(1,env.num_drones+1)}
        for drone_id in range(1,env.num_drones+1):
            reward_len = len(rewards_history[drone_id])
            for j in reversed(range(reward_len)):
                cum_rewards[drone_id][j] = rewards_history[drone_id][j] + (cum_rewards[drone_id][j+1]*gamma if j+1 < reward_len else 0)


        # Train NNs
        for drone_id in range(1, env.num_drones+1):
            values = agents[drone_id].update_critic(cum_rewards[drone_id], observations_history[drone_id])
            agents[drone_id].update_actor(actions_history[drone_id], cum_rewards[drone_id], values, observations_history[drone_id])

        sum_rewards = [sum(rewards_history[i]) for i in range(1,env.num_drones+1)]
        print(sum_rewards)

        episode_rewards.append(sum(sum_rewards))

    x = np.arange(num_episodes)
    y = np.array(episode_rewards)
      
    root.mainloop()
    plt.plot(x,y)
    plt.show()

    with open('env.pkl', 'wb') as file: 
        pickle.dump(env, file)

    for drone_id in range(1, env.num_drones+1):
        agents[drone_id].save_models("save1_test")


if __name__ == '__main__':
    main()
 