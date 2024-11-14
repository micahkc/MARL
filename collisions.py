import math

class Target:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

class Obstacle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

class Drone:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def compute_action(self, observation):
        # Given observation, return an action (e.g., move in a direction)
        # For simplicity, let's assume it returns a new position
        new_x = self.x + 1  # Example action
        new_y = self.y + 1  # Example action
        return new_x, new_y

    def updateNN(self, reward):
        # Update neural network with the given reward
        pass

class Environment:
    def __init__(self, length, width, visual=False):
        self.length = length
        self.width = width
        self.grid = [[1 for _ in range(width)] for _ in range(length)]  # grid with paths (1)
        self.targets = []
        self.obstacles = []
        self.drones = []
        self.visual = visual
        if self.visual:
            self.create_display()

    def add_target(self, target):   # we add a target as 2??

        self.targets.append(target)
        self.grid[target.x][target.y] = 2  # position

    def add_obstacle(self, obstacle):  # obstacle as a 0
        self.obstacles.append(obstacle)
        self.grid[obstacle.x][obstacle.y] = 0  

    def add_drone(self, drone):
        self.drones.append(drone)

    def step(self, actions):
        rewards = []
        observations = []
        done = False
        drones_to_remove = set()

        for i, drone in enumerate(self.drones):
            new_x, new_y = actions[i]
            if self.check_collision(new_x, new_y, drone.r):
                drones_to_remove.add(i)
            else:
                drone.x = new_x
                drone.y = new_y
                reward = self.compute_reward(drone)
                rewards.append(reward)
                observations.append((drone.x, drone.y))

        # Check for drone collisions                          ## the real juice is here. 
        for i in range(len(self.drones)):
            for j in range(i + 1, len(self.drones)):
                if self.distance(self.drones[i].x, self.drones[i].y, self.drones[j].x, self.drones[j].y) < (self.drones[i].r + self.drones[j].r):  

			# first part (before "<") calculates the euclidian distance between the 2 drones, 2nd part is the sum of the 2 radii (size) of the drones.
			# if the distance is less than the sum of the radii, consider that the drones are overlapping or colliding.  

                    drones_to_remove.add(i)
                    drones_to_remove.add(j)

        # Remove drones that collided or hit obstacles
        self.drones = [drone for i, drone in enumerate(self.drones) if i not in drones_to_remove]

        if self.visual:
            self.update_display()

        return observations, rewards, done

    def reset(self):
        # Reset environment and return the initial observation
        observations = []
        for drone in self.drones:
            drone.x, drone.y = 0, 0  # Reset positions
            observations.append((drone.x, drone.y))
        return observations

    def check_collision(self, x, y, r):     ## checking for collisions
        if x < 0 or x >= self.length or y < 0 or y >= self.width:
            return True  # Out of bounds
        if self.grid[x][y] == 0:
            return True  # Collision with obstacle
        return False

    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2) ## calculating the Euclidian distance (any other method to calculate?)

    def compute_reward(self, drone):
        # Compute reward based on drone's position and grid values
        if self.grid[drone.x][drone.y] == 1:   # return a reward for traversing a viable path
            return 1  
        elif self.grid[drone.x][drone.y] == 2:  # return a reward for doing a task 
            return 2        
        else:
            return -1  # Penalty otherwise

    def create_display(self):
        # Create visualization
        pass
        
    def update_display(self):
        # Update visualization
        pass

def wait():
    input()


def main():
    drone1 = Drone(0, 0, 1)
    drone2 = Drone(0, 0, 1)
    drone3 = Drone(0, 0, 1)
    drone4 = Drone(0, 0, 1)

    env = Environment(10, 10, visual=True)

    env.add_obstacle(Obstacle(3, 3, 1))
    env.add_target(Target(5, 5, 1))

    env.add_drone(drone1)
    env.add_drone(drone2)
    env.add_drone(drone3)
    env.add_drone(drone4)

    num_epochs = 10  # simulation runs 10 times

    for i in range(num_epochs):
        sum_rewards = 0    # initializing rewards
        observations = env.reset()  # reseting environment
        done = False  # indicating simulation isnt finished 

        while not done:  # helps to run the each simulation until over (done = True)
            actions = [drone.compute_action(obs) for drone, obs in zip(env.drones, observations)]  
            print(actions)
            wait()
            observations, rewards, done = env.step(actions)   # updating environment based on drone action
            sum_rewards += sum(rewards)  # rewards summed 

        for drone, reward in zip(env.drones, rewards):   # neural networks updated for each drone after each run
            drone.updateNN(reward)

if __name__ == "__main__":
    main()
