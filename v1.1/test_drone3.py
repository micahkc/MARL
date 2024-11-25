from tkinter import Tk
from random import randint
# Import from other python modules.
import gui
from model import Agent
def main():
    # Set learning parameters.
    num_epochs = 100
    gamma = 0.8 # Discount factor.

    env = gui.create_default_env2()

    root = Tk()
    map = gui.Map(root,env)

    # For each drone in the environment, create a learning agent.
    agents = []
    for i in range(env.num_drones):
        agents.append(Agent(i))

    # Actions are two components for each drone. These are angular acceleration and forward acceleration for this time step.
    actions = [[0,0] for x in range(env.num_drones)]
    # This is used for difference between actor and critic models.
    errors = [0 for x in range(env.num_drones)]
    
    for i in range(num_epochs):
        done = False
        sum_rewards = 0
        c=0

        observations = env.reset()
        print(f"initial observation: {observations}")
        while not done:
            # Get actions from each drone's actor policy and do these actions in the env.
            for i in range(env.num_drones):
                actions[i][0]= randint(-1,1)
                actions[i][1] = randint(-1,1)

            # #Get actions from each drone's actor policy and do these actions in the env.
            # for i in range(num_drones):
            #     actions[i] = agents[i].actor.forward(observation[i])
            
            next_observation, rewards, done = env.step(actions)
            print(rewards)
            print(next_observation)
            map.update_map(env)

            # # Update critic network (Value function).
            # for i in range(num_drones):
            #     next_value = agents[i].critic.forward(next_observation[i])
            #     target = rewards[i] + gamma*next_value
            #     errors[i] = target - agents[i].critic.forward(observation)
            #     agents[i].critic.update(observation, target)

            # # Update actor network
            # for i in range(num_drones):
            #     agents[i].actor.update(observation, errors[i], actions[i])

            observation = next_observation
            
            
            # sum_rewards += rewards
        print("need to reset")
    root.mainloop()
        
if __name__ == '__main__':
    main()