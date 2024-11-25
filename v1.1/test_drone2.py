from tkinter import Tk
from random import randint
# Import from other python modules.
import gui
from model import Agent
def main():
    # env = gui.create_env()
    env = gui.create_env()
    root = Tk()
    map = gui.Map(root,env)
    map.visual = True
    if map.visual == False:
        root.destroy()    

    # Actions are two components for each drone. These are angular acceleration and forward acceleration for this time step.
    actions = [[0,0] for x in range(env.num_drones)]
    print(f" Number of drones: {env.num_drones}")
    done = False
    sum_rewards = 0
    c=0
    while not done:
        # Get actions from each drone's actor policy and do these actions in the env.
        for i in range(env.num_drones):
            actions[i][0]= randint(-1,1)
            actions[i][1] = randint(-1,1)
        
        c=c+1
        next_observation, rewards, done = env.step(actions)
        print(rewards)
        if map.visual:
            map.update_map(env)
        
        # sum_rewards += rewards
    print("done")
    root.mainloop()
        
if __name__ == '__main__':
    main()
