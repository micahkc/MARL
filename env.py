
class Obstacle():

    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

class Target():

    def __init__(self, x, y, r, num_agents):
        self.x = x
        self.y = y
        self.r = r
        self.num_agents = num_agents

class Drone():

    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def compute_action(self, observation):
        # given observation, return an action
        return

    def updateNN(self):
        return

class Environment():

    def __init__(self, length, width, visual=False):
        self.length = length
        self.width = width
        self.targets = []
        self.obstacles = []
        self.drones = []
        self.visual = visual
        # if self.visual:
        #     # create_display()

    def add_target(self,x,y,r,num_agents):
        # need target class
        new_target = Target(x,y,r,num_agents)
        self.targets.append(new_target)
        print("Target added")

    def add_obstacle(self, x, y, r):
        # need obstacle class
        new_obstacle = Obstacle(x,y,r)
        self.obstacles.append(new_obstacle)
        print("Obstacle added")

    def add_drone(self, x, y):
        new_drone = Drone(x,y)
        self.drones.append(new_drone)
        print("Drone added")

    def step(self, action):
        # updates the enviroment given new position and velocities (check for collisions)
        # return rewards observations for given action 
        if self.visual == True:
            self.update_display()

    def reset(self):
        # reset environment and return the observation
        return

    def create_display():
        # make display
        return
        
    def update_display():
        return
        
    
# Collision detection. Aniket! 
# Obstacles [(x,y,r), (x,y,r)...]
# Targets [(x,y,r), (x,y,r)...]
# Drones [(x,y,r), (x,y,r)...]
# Drones change, others do not 

    
# def main():
#     # Create input gui for info about drones and environment.

#     num_epochs = 10

#     drone1 = Drone()
#     drone2 = Drone()
#     drone3 = Drone()
#     drone4 = Drone()

#     env = Environment([],[])

#     for i in range(num_epochs):
#         sum_rewards = 0
#         observation = env.reset()
#         while not done:
#             action1 = drone1.compute_action(observation)
#             action2 = drone2.compute_action(observation)
#             action3 = drone3.compute_action(observation)
#             action4 = drone4.compute_action(observation)
#             action = [action1, action2, action3, action4]
#             observation, reward, done = env.step(action)
#             sum_rewards += reward

#         drone1.updateNN(reward[0])
#         drone2.updateNN(reward[1])
#         drone3.updateNN(reward[2])
#         drone4.updateNN(reward[3])

# main()