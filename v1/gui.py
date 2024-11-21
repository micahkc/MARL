from tkinter import *
from env import Environment
import time
import random as rand
# from project_Micah_update_10-11 import *

#--------------------Colours------------------------
BEIGE = '#ECDBBA'
RED = '#C84B31'
NAVY = '#2D4263'
BLACK = '#191919'
#--------------------------------------------------

##############################################################################################################
#-------------------- BUTTON FUNCTIONS -----------------------------------------------------------------------#
def modify_env():
    global env, env_label, target_count_label, obstacle_count_label, drone_count_label, length_entry, width_entry

    env.length = int(length_entry.get())
    env.width = int(width_entry.get())
    env.targets = []
    env.drones = []
    env.obstacles = []
    print(f'A {env.width}x{env.length} environment has been created')

    env_label.config(text=f"{env.width}x{env.length} Environment Created")

    target_count_label.config(text="0 Target(s)")
    obstacle_count_label.config(text="0 Obstacle(s)")
    drone_count_label.config(text="0 Drone(s)")

def add_target():

    global env, target_count_label, root

    target_win = Toplevel(root)
    target_win.title("Add Target")
    target_win.geometry("300x200")
    target_win.config(padx=10, pady=10)

    # labels
    x_label = Label(target_win, text="x Position")
    x_label.place(x=0,y=0)

    y_label = Label(target_win, text="y Position")
    y_label.place(x=0,y=20)

    radius_label = Label(target_win, text="Radius")
    radius_label.place(x=0,y=40)

    num_label = Label(target_win, text="Num Agents")
    num_label.place(x=0,y=60)

    # entries
    x_entry = Entry(target_win, width=15)
    x_entry.place(x=80,y=0)

    y_entry = Entry(target_win, width=15)
    y_entry.place(x=80,y=20)

    radius_entry = Entry(target_win, width=15)
    radius_entry.place(x=80,y=40)

    num_entry = Entry(target_win, width=15)
    num_entry.place(x=80,y=60)

     # change the target count label
    def change_num_targets_label():
        num_targets = len(env.targets)
        target_count_label.config(text=f"{num_targets} Target(s)")

    # add_target button
    button = Button(target_win, text="Add", command= lambda: [env.add_target(float(x_entry.get()), 
                                                                             float(y_entry.get()), 
                                                                             float(radius_entry.get()), 
                                                                             int(num_entry.get())), 
                                                              change_num_targets_label(), 
                                                              target_win.destroy()])
    button.place(x=10, y=90)

   
    

def add_obstacle():
        
    global env, obstacle_count_label, root

    obstacle_win = Toplevel(root)
    obstacle_win.title("Add Obstacle")
    obstacle_win.geometry("300x200")
    obstacle_win.config(padx=10, pady=10)

    # labels
    x_label = Label(obstacle_win, text="x Position")
    x_label.place(x=0,y=0)

    y_label = Label(obstacle_win, text="y Position")
    y_label.place(x=0,y=20)

    radius_label = Label(obstacle_win, text="Radius")
    radius_label.place(x=0,y=40)

    # entries
    x_entry = Entry(obstacle_win, width=15)
    x_entry.place(x=80,y=0)

    y_entry = Entry(obstacle_win, width=15)
    y_entry.place(x=80,y=20)

    radius_entry = Entry(obstacle_win, width=15)
    radius_entry.place(x=80,y=40)

    # change the target count label
    def change_num_obstacles_label():
        num_obs = len(env.obstacles)
        obstacle_count_label.config(text=f"{num_obs} Obstacle(s)")

    # add_obstacle button
    button = Button(obstacle_win, text="Add", command=lambda: [env.add_obstacle(float(x_entry.get()),float(y_entry.get()),float(radius_entry.get())), 
                                                               change_num_obstacles_label(), 
                                                               obstacle_win.destroy()])
    button.place(x=10, y=90)

def add_drone():
        
    global env, root, drone_count_label

    drone_win = Toplevel(root)
    drone_win.title("Add Drone")
    drone_win.geometry("300x200")
    drone_win.config(padx=10, pady=10)

    # labels
    x_label = Label(drone_win, text="Starting x Position")
    x_label.place(x=0,y=0)

    y_label = Label(drone_win, text="Starting y Position")
    y_label.place(x=0,y=20)

    # entries
    x_entry = Entry(drone_win, width=15)
    x_entry.place(x=105,y=0)

    y_entry = Entry(drone_win, width=15)
    y_entry.place(x=105,y=20)

    # change the target count label
    def change_num_drones_label():
        num_drones = len(env.drones)
        drone_count_label.config(text=f"{num_drones} Drone(s)")

    # add_drone button
    button = Button(drone_win, text="Add", command=lambda: [env.add_drone(float(x_entry.get()),float(y_entry.get())), 
                                                            change_num_drones_label(), 
                                                            drone_win.destroy()])
    button.place(x=10, y=90)


def draw_map():
    global env, root

    # create map window
    map_win = Toplevel(root, bg=BEIGE)
    map_win.title(f"Map {env.width}x{env.length}")
    win_size = 800

    # scale the map acutal length to window size
    map_size = max([env.length, env.width])
    scale = win_size/map_size
    scaled_length = env.length*scale
    scaled_width = env.width*scale
    win_length = int(round(scaled_length))
    win_width = int(round(scaled_width))
    map_win.minsize(width=win_width, height=win_length)

    # create canvas for the map
    map_canvas = Canvas(map_win, width=win_width, height=win_length, bg=BEIGE)
    map_canvas.pack()
    
    # add targets to the map
    for target in env.targets:
        x = target.x *scale
        y = target.y *scale
        r = target.r *scale
        map_canvas.create_oval(x-r, y-r, x+r, y+r, outline = "black", fill = RED, width = 2)

    # add all obstacles to the map
    for obs in env.obstacles:
        x = obs.x *scale
        y = obs.y *scale
        r = obs.r *scale
        map_canvas.create_oval(x-r, y-r, x+r, y+r, outline = "black", fill = NAVY, width = 2)

    # add all drones to the map
    for drone in env.drones:
        r = 7 # square half width
        x = drone.x *scale
        y = drone.y *scale
        map_canvas.create_rectangle(x-r, y-r, x+r, y+r, outline = "black", fill = BLACK, width = 2)

    
#-------------------- END OF BUTTON FUNCTIONS -----------------------------------------------------------------------#
######################################################################################################################

#---------------------MAIN LOOP FUNCTION FOR GUI---------------------------------------------------------------------#

def create_env():
    global env, env_label, target_count_label, obstacle_count_label, drone_count_label, length_entry, width_entry
    global root, target_count_label, obstacle_count_label, drone_count_label

    root = Tk()
    root.title('UAV SWARM')
    root.minsize(width=300, height=300)
    root.config(padx=10, pady=10)

    # create a default environment
    env = Environment(1000, 1000)

    # environment label
    env_label = Label(text=f"{env.length}x{env.width} (Default) Environment Created")
    env_label.place(x=0,y=0)

    # length/width label
    length_label = Label(text="Length")
    length_label.place(x=0,y=20)
    width_label = Label(text="Width")
    width_label.place(x=0,y=40)

    # ENTER LENGTH AND WIDTH OF THE MAP
    length_entry=Entry(width=30)
    length_entry.place(x=50,y=20)
    width_entry = Entry(width=30)
    width_entry.place(x=50,y=40)

    # create environment button
    button = Button(text="Create Environment", command=modify_env)
    button.place(x=0,y=60)

    # create target button
    create_target_button = Button(text="Add Target", command=add_target)
    create_target_button.place(x=0,y=100)

    # target counter label
    target_count_label = Label(text='0 Target(s)')
    target_count_label.place(x=100, y=100)

    # create obstacle button
    create_obstacle_button = Button(text="Add Obstacle", command=add_obstacle)
    create_obstacle_button.place(x=0,y=140)

    # obstacle counter label
    obstacle_count_label = Label(text='0 Obstacle(s)')
    obstacle_count_label.place(x=100,y=140)

    # add drone button
    create_drone_button = Button(text="Add Drone", command=add_drone)
    create_drone_button.place(x=0,y=180)

    # drone counter label
    drone_count_label = Label(text='0 Drone(s)')
    drone_count_label.place(x=100,y=180)

    # run button
    run_button = Button(text='Run', command=root.destroy)
    run_button.place(x=200,y=140)

    # draw map button
    draw_map_button = Button(text="Draw Map", command=draw_map)
    draw_map_button.place(x=200,y=180)


    root.mainloop()

    return env


def create_default_env():
    global env, env_label, target_count_label, obstacle_count_label, drone_count_label, length_entry, width_entry
    global root, target_count_label, obstacle_count_label, drone_count_label


    # root = Tk()

    # create a default environment
    env = Environment(1000, 1000)

    drone_params = [[250, 250], [250, 500], [250, 750]]
    for param in drone_params:
        env.add_drone(param[0], param[1])


    obstacle_params = [[500, 250, 50], [500, 500, 50], [500, 750, 50]]
    for param in obstacle_params:
        env.add_obstacle(param[0], param[1], param[2])

    target_params = [[750, 250, 50, 2], [750, 500, 50, 2], [750, 750, 50, 2]]
    for param in target_params:
        env.add_target(param[0], param[1], param[2], param[3])

    # draw_map()
    # root.mainloop()

    return env

# def update_map(env, map_win):
#     # create map window
#     map_win.title(f"Map {env.width}x{env.length}")
#     win_size = 800

#     # scale the map acutal length to window size
#     map_size = max([env.length, env.width])
#     scale = win_size/map_size
#     scaled_length = env.length*scale
#     scaled_width = env.width*scale
#     win_length = int(round(scaled_length))
#     win_width = int(round(scaled_width))
#     map_win.minsize(width=win_width, height=win_length)

#     # create canvas for the map
#     map_canvas = Canvas(map_win, width=win_width, height=win_length, bg=BEIGE)
#     map_canvas.pack()
    
#     # add targets to the map
#     for target in env.targets:
#         x = target.x *scale
#         y = target.y *scale
#         r = target.r *scale
#         map_canvas.create_oval(x-r, y-r, x+r, y+r, outline = "black", fill = RED, width = 2)

#     # add all obstacles to the map
#     for obs in env.obstacles:
#         x = obs.x *scale
#         y = obs.y *scale
#         r = obs.r *scale
#         map_canvas.create_oval(x-r, y-r, x+r, y+r, outline = "black", fill = NAVY, width = 2)

#     # add all drones to the map
#     for drone in env.drones:
#         r = 7 # square half width
#         x = drone.x *scale
#         y = drone.y *scale
#         map_canvas.create_rectangle(x-r, y-r, x+r, y+r, outline = "black", fill = BLACK, width = 2)



class Map:
    def __init__(self, root, env):
        self.root = root
        self.root.title("Map Animation")

        # scale the map acutal length to window size
        win_size = 800
        map_size = max([env.length, env.width])
        self.scale = win_size/map_size
        scaled_length = env.length*self.scale
        scaled_width = env.width*self.scale
        win_length = int(round(scaled_length))
        win_width = int(round(scaled_width))
        root.minsize(width=win_width, height=win_length)

        # create canvas for the map
        self.canvas = Canvas(root, width=win_width, height=win_length, bg=BEIGE)
        self.canvas.pack()

        self.targets = []
        for target in env.targets:
            x = target.x *self.scale
            y = target.y *self.scale
            r = target.r *self.scale
            oval = self.canvas.create_oval(x-r, y-r, x+r, y+r, outline = "black", fill = RED, width = 2)
            self.targets.append(oval)

        self.obstacles = []
        # add all obstacles to the map
        for obs in env.obstacles:
            x = obs.x *self.scale
            y = obs.y *self.scale
            r = obs.r *self.scale
            oval = self.canvas.create_oval(x-r, y-r, x+r, y+r, outline = "black", fill = NAVY, width = 2)
            self.obstacles.append(oval)
        
        self.drones = []
        # add all drones to the map
        for drone in env.drones:
            r = 7 # square half width
            x = drone.x *self.scale
            y = drone.y *self.scale
            rectangle = self.canvas.create_rectangle(x-r, y-r, x+r, y+r, outline = "black", fill = BLACK, width = 2)
            self.drones.append(rectangle)


    def update_map(self, env):
        # Update the positions of the objects
        for rect in self.drones:
            self.canvas.delete(rect)
        self.drones = []
        for drone in env.drones:
            r = 7 # square half width
            x = drone.x *self.scale
            y = drone.y *self.scale
            rectangle = self.canvas.create_rectangle(x-r, y-r, x+r, y+r, outline = "black", fill = BLACK, width = 2)
            self.drones.append(rectangle)

        self.root.after(100)
        self.canvas.update()
    

if __name__ == "__main__":
    root = Tk()
    env = create_default_env()
    map = Map(root,env)

    for j in range(1000):
        for drone in env.drones:
            drone.x = drone.x + rand.randint(-10,10)
            drone.y = drone.y + rand.randint(-10,10)
        map.update_map(env)

    root.mainloop()

