# Import from other python modules.
import gui
from model import Agent
def main():
    # Set learning parameters.
    num_epochs = 100
    gamma = 0.8 # Discount factor.

    # Create Environment including obstacles, targets, and drones, using GUI.
    # env = gui.create_env()
    env = gui.create_default_env()
    print("Environment Created")

    # For each drone in the environment, create a learning agent.
    agents = []
    num_drones = len(env.drones)
    for i in range(num_drones):
        agents.append(Agent(i))

    # Actions are two components for each drone. These are angular acceleration and forward acceleration for this time step.
    actions = [0 for x in range(num_drones)]

    while not done:
        # Get actions from each drone's actor policy and do these actions in the env.
        for i in range(num_drones):
            actions[i] = [1, 1]

        next_observation, rewards, done = env.step(actions)
        
        sum_rewards += rewards
        

        observation = next_observation
        
if __name__ == '__main__':
    main()
