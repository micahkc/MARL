
from math import sqrt, atan2

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
    
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id
        self.v_x = 0
        self.v_y = 0
        # Radius fo drone.
        self.r = 20
        self.scan_radius = 50
        self.active = True
        
class Environment():

    def __init__(self, length, width):
        self.length = length
        self.width = width
        self.num_drones = 0
        self.targets = []
        self.obstacles = []
        self.drones = []

        # Use control input for this many seconds.
        self.ctrl_rate = 0.5 
        

    def add_target(self,x,y,r,num_agents):
        # need target class
        new_target = Target(x,y,r,num_agents)
        self.targets.append(new_target)

    def add_obstacle(self, x, y, r):
        # need obstacle class
        new_obstacle = Obstacle(x,y,r)
        self.obstacles.append(new_obstacle)

    def add_drone(self, x, y):
        self.num_drones += 1
        new_drone = Drone(x,y, self.num_drones)
        self.drones.append(new_drone)

    def check_obstacle_collision(self, x, y, radius):
        for obstacle in self.obstacles:
            if self.distance(x, y, obstacle.x, obstacle.y) < (radius + obstacle.r):
                return True
        return False
    
    def check_drone_collision(self, x, y, radius, id):
        for drone in self.drones:
            if self.distance(x, y, drone.x, drone.y) < (radius + drone.r) and drone.active and id != drone.id:
                return True
        return False

    def distance(self, x1, y1, x2, y2):
        return sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def step(self, actions):
        # updates the enviroment given new control input (check for collisions)
        # return rewards, observations, completion for given action 
        # Update drone positions.
        for i,action in enumerate(actions):
            prev_x = self.drones[i].x
            prev_y = self.drones[i].y
            prev_v_x = self.drones[i].v_x
            prev_v_y = self.drones[i].v_y

            # Get acceleration from control inputs (actions).
            a_x = action[0]
            a_y = action[1]
            
            # Obtain new velocity by integrating acceleration for time step.
            v_x = a_x*self.ctrl_rate + prev_v_x
            v_y = a_y*self.ctrl_rate + prev_v_y

            # Obtain new position by integrating acceleration and velocity for time step.
            x = 0.5*a_x*(self.ctrl_rate**2) + v_x*self.ctrl_rate + prev_x
            y = 0.5*a_y*(self.ctrl_rate**2) + v_y*self.ctrl_rate + prev_y

            # Save parameters to drone.
            self.drones[i].x = x
            self.drones[i].y = y
            self.drones[i].v_x = x
            self.drones[i].v_y = y

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
                elif self.check_drone_collision(drone.x, drone.y, drone.radius, drone.id):
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
        # Observation is [type of observation, nearest position relative to drone (x,y)]
        # Options for type of observation are 0: no detection, 1: target detection, 2: obstacle detection, 3: drone detection.
        observations = []
        for i, drone in enumerate(self.drones):
            if drone.active:
                scan_results = []
                # Check for obstacles within scan.
                for obstacle in self.obstacles:
                    d = self.distance(drone.x, drone.y, obstacle.x, obstacle.y)
                    if d < drone.scan_radius:
                        delta_y = obstacle.y - drone.y
                        delta_x = obstacle.x - drone.x
                        scan_results.append([1, delta_x, delta_y])

                # Check for other drones within scan.
                for drone2 in self.drones:
                    d = self.distance(drone.x, drone.y, drone2.x, drone2.y)
                    if d < drone.scan_radius:
                        delta_y = drone2.y - drone.y
                        delta_x = drone2.x - drone.x
                        scan_results.append([2, delta_x, delta_y])

                # Check for targets within scan.
                for target in self.targets:
                    d = self.distance(drone.x, drone.y, target.x, target.y)
                    if d < drone.scan_radius:
                        delta_y = target.y - drone.y
                        delta_x = target.x - drone.x
                        scan_results.append([3, delta_x, delta_y])

            scan_type = [0 for i in range(4)]
            scan_coordinates = [[1000,1000] for i in range(8)]
            for scan, j in enumerate(scan_results):
                idx, delta_x, delta_y = scan
                scan_type[idx] += 1
                scan_coordinates[j] = [delta_x, delta_y]
            drone_observation = scan_type + scan_coordinates
            observations.append(drone_observation)


        
        return observations, rewards, done


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