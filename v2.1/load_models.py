from tkinter import Tk
from random import randint
import torch
import numpy as np
import matplotlib.pyplot as plt
# Import from other python modules.
import gui
from model import Agent
def main():
    # Set learning parameters.
    num_episodes = 100
    gamma = 0.8 # Discount factor.

    env = gui.create_default_env3()

    root = Tk()
    map = gui.Map(root,env)
    map.visual = True
    if not map.visual:
        root.destroy()

    # For each drone in the environment, create a learning agent (includes both policy and value nns).
    agents = {}
    for drone_id in range(1, env.num_drones+1):
        agents[drone_id] = Agent(drone_id)
        agents[drone_id].create_models()
        agents[drone_id].load_models("save1_test")
    
    episode_rewards = []
    # Training loop
    for i in range(num_episodes):
        # Create empty history for observations, actions, and rewards for this episoide.
        done = False

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
                print(actions)
                
            # Implement these actions in the env.
            next_observations, rewards, done = env.step(actions)
            
            # Update map visualization for this step if required.
            if map.visual:
                map.update_map(env)

if __name__ == '__main__':
    main()
