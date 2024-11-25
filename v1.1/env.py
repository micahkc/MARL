
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
        self.active = True

    def distance(self, obj):
        return sqrt((self.x - obj.x)**2 + (self.y - obj.y)**2)

    def check_target_success(self,env):
        # counter for number of drones on target
        num_drones_on_target = 0
        drones_on_target = []

        # check if each drone is on the target or not
        # only consider active drones
        for drone in env.drones:
            if drone.active:
                if self.distance(drone) < self.r + drone.r:
                    num_drones_on_target += 1
                    drones_on_target.append(drone)
        
        # if the number of drones on target matches or exceeds the required number,
        # then return true.
        if num_drones_on_target >= self.num_agents:
            return True, drones_on_target
        else:
            return False, drones_on_target

class Drone():
    
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id
        self.v_x = 0
        self.v_y = 0
        # Radius for drone.
        self.r = 10
        self.scan_radius = 150
        self.active = True

    def distance(self, obj):
        return sqrt((self.x - obj.x)**2 + (self.y - obj.y)**2)

    def get_observation(self, env):

        # check for in view obstacles
        in_view_obstacles = []
        for obstacle in env.obstacles:
            if self.distance(obstacle) < (self.scan_radius+obstacle.r):
                in_view_obstacles.append(obstacle)

        # check for in view drones
        in_view_drones = []
        for drone in env.drones:
            if self.id != drone.id:
                if drone.active:
                    if self.distance(drone) < (self.scan_radius+drone.r):
                        in_view_drones.append(drone)

        # check for in view targets
        in_view_targets = []
        for target in env.targets:
            if target.active:
                if self.distance(target) < (self.scan_radius+target.r):
                    in_view_targets.append(target)
        
        return (in_view_drones, in_view_targets, in_view_obstacles)
        
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
    
    def get_observations(self):
        # Observation is in format [drone coordinates, drone velocity, coordinates of drones in view, coordinates of obstacles in view, coordinates of targets in view]
        observations = {}
        for i,drone in enumerate(self.drones):
            drone_coord = []
            drone_vel = []
            nearby_drones = [[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
            nearby_obstacles = [[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
            nearby_targets = [[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
            # Obtain drone coordinates.
            drone_coord.append([drone.x, drone.y])
            drone_vel.append([drone.v_x, drone.v_y])

            in_view_drones, in_view_targets, in_view_obstacles = drone.get_observation(self)
            
            for j,d in enumerate(in_view_drones):
                print("Drone detected")
                nearby_drones[j] = [d.x, d.y]

            for j,o in enumerate(in_view_obstacles):
                print("obstacle detected")
                nearby_obstacles[j] = [o.x, o.y]

            for j,t in enumerate(in_view_targets):
                print("target detected")
                print([t.x, t.y])
                nearby_targets[j] = [t.x, t.y]

            obs = [drone_coord, drone_vel, nearby_drones, nearby_obstacles, nearby_targets]
            observations[drone.id] = obs
        return observations

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

            # Save parameters to drone if active.
            if self.drones[i].active:
                self.drones[i].x = x
                self.drones[i].y = y
                self.drones[i].v_x = v_x
                self.drones[i].v_y = v_y
        # Negative reward for going off screen

        # Check for collisions now that drones are in new positions.
        drones_to_remove = set()
        # rewards = [0 for x in range(self.num_drones)]
        rewards = {drone.id:0 for drone in self.drones}
        for i, drone in enumerate(self.drones):
            if drone.active:
                # Check for drone going off screen.
                if drone.x < 0 or drone.x > self.width or drone.y < 0 or drone.y > self.length:
                    drones_to_remove.add(i)
                                 
                # Check for obstacle collision.
                elif self.check_obstacle_collision(drone.x, drone.y, drone.r):
                    drones_to_remove.add(i)
                
                # Check for drone-drone collision.
                # Compare with other drones, not itself.
                elif self.check_drone_collision(drone.x, drone.y, drone.r, drone.id):
                    drones_to_remove.add(i)

                # Check for target acheivement.
                else:
                    continue
                    for target in self.targets:
                        if target.active:
                            if self.distance(drone.x, drone.y, target.x, target.y) < (drone.r + target.r):
                                print("Mission Accomplished")
                                target.active = False
                                rewards[drone.id] += 1
                                # self.targets.remove(target)
                                
                # Add negative reward based on proximity to the edge.
                proximity_x = 0
                proximity_y = 0
                proximity = 0
                if drone.x < 0 or drone.x > self.width:
                    proximity_x = min(abs(drone.x), abs(self.width - drone.x))
                    if proximity_x > 100:
                        self.drone[i].active = False
                    
                if drone.y < 0 or drone.y > self.length:
                    proximity_y = min(abs(drone.y), abs(self.length - drone.y))
                    if proximity_y > 100:
                        self.drone[i].active = False
                                   
                proximity += proximity_x + proximity_y
                rewards[i] -= (proximity) * 0.1
                
        # loop through each target to check if target is accomplished
        for target in self.targets:
            if target.active:
                target_success, drones_on_target = target.check_target_success(self)
                if target_success:
                    print("Mission Accomplished")
                    target.active = False
                    for drone in drones_on_target:
                        rewards[drone.id] += 1
                        # give the drone a reward
                        continue

        # Remove collided drones.
        for i in drones_to_remove:
            self.drones[i].active = False
        
        # Check if all drones are inactive (game over).
        active_drones = [drone for drone in self.drones if drone.active]
        done = False
        if len(active_drones) < 1:
            done = True
            
        # Obtain next observation for the drone.
        # Observation is [drone position, drone speed, type of observation, nearest position relative to drone (x,y)]
        observations = self.get_observations()
        
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