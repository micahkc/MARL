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

    # Train.
    for i in range(num_epochs):
        sum_rewards = 0
        # Observation is [type of observation, position relative to drone (x,y)]
        # Options for type of observation are 0: no detection, 1: target detection, 2: obstacle detection, 3: drone detection, 4: multiple detection
        observation = env.reset()
        # Actions are two components for each drone. These are angular acceleration and forward acceleration for this time step.
        actions = [0 for x in range(num_drones)]
        # This is used for difference between actor and critic models.
        errors = [0 for x in range(num_drones)]
        while not done:
            # Get actions from each drone's actor policy and do these actions in the env.
            for i in range(num_drones):
                actions[i] = agents[i].actor.forward(observation[i])

            next_observation, rewards, done = env.step(actions)
            
            sum_rewards += rewards
            
            # Update critic network (Value function).
            for i in range(num_drones):
                next_value = agents[i].critic.forward(next_observation[i])
                target = rewards[i] + gamma*next_value
                errors[i] = target - agents[i].critic.forward(observation)
                agents[i].critic.update(observation, target)

            # Update actor network
            for i in range(num_drones):
                agents[i].actor.update(observation, errors[i], actions[i])

            observation = next_observation
        
if __name__ == '__main__':
    main()
