   
def main():
    # Create input gui for info about drones and environment.

    drones = []
    num_drones = 4
    num_epochs = 100

    # Create drones.
    for i in range(num_drones):
        drones.append(Drone())

    # Create Environment.
    env = Environment([],[])

    # Train.
    for i in range(num_epochs):
        sum_rewards = 0
        observation = env.reset()
        errors = [0 for x in range(num_drones)]
        actions = [0 for x in range(num_drones)]
        while not done:
            # Get actions from each drone's actor policy and do these actions in the env.
            for i in range(num_drones):
                actions[i] = drones[i].actor.forward(observation[i])

            next_observation, rewards, done = env.step(actions)
            
            sum_rewards += rewards
            
            # Update critic network (Value function).
            for i in range(num_drones):
                next_value = drones[i].critic.forward(next_observation[i])
                target = rewards[i] + gamma*next_value
                errors[i] = target - drones[i].critic.forward(observation)
                drones[i].critic.update(observation, target)

            # Update actor network
            for i in range(num_drones):
                drones[i].actor.update(observation, errors[i], actions[i])

            observation = next_observation
        
