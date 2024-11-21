
from math import sin, cos
import math

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
        # Local coordinate system.
        self.r = 0
        self.theta = 0
        self.v_r = 0
        self.v_theta = 0
        self.active = True
        self.radius = 20
        


class Environment():

    def __init__(self, length, width, visual=False):
        self.length = length
        self.width = width
        self.targets = []
        self.obstacles = []
        self.drones = []
        self.visual = visual

        # Use control input for this many seconds.
        self.ctrl_rate = 0.5 
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

    def check_obstacle_collision(self, x, y, radius):
        for obstacle in self.obstacles:
            if self.distance(x, y, obstacle.x, obstacle.y) < (radius + obstacle.r):
                return True
        return False
    
    def check_drone_collision(self, x, y, radius):
        for drone in self.drones:
            if self.distance(x, y, drone.x, drone.y) < (radius + drone.radius) and drone.active:
                return True
        return False

    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def step(self, actions):
        """
        # updates the enviroment given new control input (check for collisions)
        # return rewards, observations, completion for given action 


        # Update drone positions.
        for i,action in enumerate(actions):
            prev_x = self.drones[i].x
            prev_y = self.drones[i].y
            prev_theta = self.drones[i].theta
            prev_v_theta = self.drones[i].v_theta
            prev_v_r = self.drones[i].v_r

            # Get acceleration from control inputs (actions).
            a_theta = action[0]
            a_r = action[1]
            
            # Obtain new velocity by integrating acceleration for time step.
            v_theta = a_theta*self.ctrl_rate + prev_v_theta
            v_r = a_r*self.ctrl_rate + prev_v_r

            # Obtain new position by integrating acceleration and velocity for time step.
            theta = 0.5*a_theta*(self.ctrl_rate**2) + prev_v_theta*self.ctrl_rate + prev_theta
            r = 0.5*a_r*(self.ctrl_rate**2) + prev_v_r*self.ctrl_rate

            # Convert new position to global frame.
            x = prev_x + r*cos(theta)
            y = prev_y + r*sin(theta)

            # Save parameters to drone.
            self.drones[i].x = x
            self.drones[i].y = y
            self.drones[i].theta = theta
            self.drones[i].v_theta = v_theta
            self.drones[i].v_r = v_r

        # Check for collisions now that drones are in new positions.
        drones_to_remove = set()
        rewards = [0 for 0 in len(self.drones)]
        for i, drone in enumerate(self.drones):
            if drone.active:
                # Check for drone going off screen.
                if drone.x < 0 or drone.x > self.width or drone.y < 0 or drone.y > self.length:
                    drones_to_remove.add(i)

                # Check for obstacle collision.
                elif self.check_obstacle_collision(drone.x, drone.y, drone.radius):
                    drones_to_remove.add(i)
                
                # Check for drone-drone collision.
                # Compare with other drones, not itself.
                elif self.check_drone_collision(drone.x, drone.y, drone.radius):
                    drones_to_remove.add(i)

                # Check for target acheivement.
                else:
                    for target in self.targets:
                        if self.distance(drone.x, drone.y, target.x, target.y) < (drone.radius + target.r):
                            rewards[i] += 1
                            self.targets.remove(target)

        # Remove collided drones.
        for i in drones_to_remove:
            self.drones[i].active = False
        
        # Check if all drones are inactive (game over).
        active_drones = [drone for drone in self.drones if drone.active]
        done = False
        if len(active_drones < 1):
            done = True
        
        # Obtain next observation for the drone.
        # Observation is [type of observation, nearest position relative to drone (r,theta)]
        # Options for type of observation are 0: no detection, 1: target detection, 2: obstacle detection, 3: drone detection.
        
        return next_observation, rewards, done
        """

    def reset(self):
        # reset environment and return the observation
        return

    def get_params():
        # Returns all necessary parameters for visualization.
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