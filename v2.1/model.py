import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Agent():
    def __init__(self, n):
        self.id = n

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()

            self.hidden1 = nn.Linear(28, 28)
            self.hidden2 = nn.Linear(28, 16)
            self.hidden3 = nn.Linear(16, 8)
            self.output= nn.Linear(8,2)

            logstds_param = nn.Parameter(torch.full((2,), -0.8))
            self.register_parameter("logstds", logstds_param)

        def forward(self, s):
            outs = self.hidden1(s)
            outs = F.tanh(outs)
            outs = self.hidden2(outs)
            outs = F.tanh(outs)
            outs = self.hidden3(outs)
            outs = F.tanh(outs)
            means = self.output(outs)
            

            stds = torch.clamp(self.logstds.exp(), 1e-3, 50)

            return torch.distributions.Normal(means, stds)
    
    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden1 = nn.Linear(28, 28)
            self.hidden2 = nn.Linear(28, 16)
            self.hidden3 = nn.Linear(16, 4)
            self.output= nn.Linear(4,1)

        def forward(self, s):
            outs = self.hidden1(s)
            outs = F.tanh(outs)
            outs = self.hidden2(outs)
            outs = F.tanh(outs)
            outs = self.hidden3(outs)
            outs = F.tanh(outs)
            value = self.output(outs)
            return value
    
    def create_models(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = self.Actor().to(device)
        self.critic = self.Critic().to(device)

        self.optimizerA = optim.Adam(self.actor.parameters())
        self.optimizerC = optim.Adam(self.critic.parameters())

    def update_critic(self, cum_rewards, observations):
        self.optimizerC.zero_grad()
        cum_rewards = torch.tensor(cum_rewards, dtype=torch.float)
        observations = torch.tensor(observations, dtype = torch.float)
        values = self.critic.forward(observations)
        values = values.squeeze(dim=1)
        loss = F.mse_loss(values, cum_rewards, reduction="none")
        loss.sum().backward()
        self.optimizerC.step()
        return values
    
    def save_models(self,path):
        torch.save(self.critic.state_dict(), f"{path}_critic_{self.id}")
        torch.save(self.actor.state_dict(), f"{path}_actor_{self.id}")

    def load_models(self, path):
        self.critic.load_state_dict(torch.load(f"{path}_critic_{self.id}", weights_only=True))
        self.actor.load_state_dict(torch.load(f"{path}_actor_{self.id}", weights_only=True))

    def update_actor(self, actions, cum_rewards, values, observations):
        self.optimizerA.zero_grad()
        actions = torch.tensor(actions, dtype=torch.float)
        observations = torch.tensor(observations, dtype = torch.float)
        cum_rewards = torch.tensor(cum_rewards, dtype=torch.float)
        advantage = cum_rewards - values
        norm_dists = self.actor.forward(observations)
        logs_probs = norm_dists.log_prob(actions)
        advantage = torch.unsqueeze(advantage,1)
        #entropy = norm_dists.entropy().mean()
        actor_loss = (-logs_probs*advantage.detach()).mean()# - entropy*self.entropy_beta
        actor_loss.backward()
        self.optimizerA.step()
        
    