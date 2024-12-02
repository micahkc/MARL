import tkinter as tk
import random
import math

class Target:
    def __init__(self, canvas, x, y, r):
        self.x = x
        self.y = y
        self.r = r
        self.canvas = canvas
        self.id = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="green")

class Obstacle:
    def __init__(self, canvas, x, y, r):
        self.x = x
        self.y = y
        self.r = r
        self.canvas = canvas
        self.id = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="red")

class Drone:
    def __init__(self, canvas, x, y, r):
        self.x = x
        self.y = y
        self.r = r
        self.active = True
        self.canvas = canvas
        self.id = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="blue")

    def compute_action(self):
        if self.active:
            # Increase speed by adjusting the magnitude of movement
            new_x = self.x + random.uniform(-1, 1) * 10
            new_y = self.y + random.uniform(-1, 1) * 10
            return new_x, new_y
        return self.x, self.y

class Environment:
    def __init__(self, length, width):
        self.length = length
        self.width = width
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=width, height=length)
        self.canvas.pack()
        self.targets = []
        self.obstacles = []
        self.drones = []
        self.score = 0

    def add_target(self, target):
        self.targets.append(target)

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def add_drone(self, drone):
        self.drones.append(drone)

    def check_collision(self, x, y, radius):
        for obstacle in self.obstacles:
            if self.distance(x, y, obstacle.x, obstacle.y) < (radius + obstacle.r):
                return True
        return False

    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def step(self):
        drones_to_remove = set()
        for i, drone in enumerate(self.drones):
            if drone.active:
                new_x, new_y = drone.compute_action()
                if new_x < 0 or new_x > self.width or new_y < 0 or new_y > self.length or self.check_collision(new_x, new_y, drone.r):
                    drones_to_remove.add(i)
                else:
                    self.canvas.move(drone.id, new_x - drone.x, new_y - drone.y)
                    drone.x, drone.y = new_x, new_y
                    for target in self.targets:
                        if self.distance(drone.x, drone.y, target.x, target.y) < (drone.r + target.r):
                            self.score += 1
                            self.targets.remove(target)
                            self.canvas.delete(target.id)
            else:
                drones_to_remove.add(i)

        # Check for drone collisions
        for i in range(len(self.drones)):
            for j in range(i + 1, len(self.drones)):
                if self.drones[i].active and self.drones[j].active:
                    if self.distance(self.drones[i].x, self.drones[i].y, self.drones[j].x, self.drones[j].y) < (self.drones[i].r + self.drones[j].r):
                        drones_to_remove.add(i)
                        drones_to_remove.add(j)

        # Remove collided drones
        for i in drones_to_remove:
            self.drones[i].active = False
            self.canvas.itemconfig(self.drones[i].id, fill="grey")

        self.root.after(50, self.step)  # Faster updates for quicker movement

    def start(self):
        self.root.after(50, self.step)
        self.root.mainloop()

def main():
    env = Environment(1000, 1000)

    # Randomly place obstacles
    for _ in range(10):
        x, y, r = random.randint(50, 950), random.randint(50, 950), 20
        env.add_obstacle(Obstacle(env.canvas, x, y, r))

    # Randomly place targets
    for _ in range(5):
        x, y, r = random.randint(50, 950), random.randint(50, 950), 10
        env.add_target(Target(env.canvas, x, y, r))

    # Add drones
    drones = [Drone(env.canvas, random.randint(50, 950), random.randint(50, 950), 15) for _ in range(4)]
    for drone in drones:
        env.add_drone(drone)

    env.start()

if __name__ == "__main__":
    main()


