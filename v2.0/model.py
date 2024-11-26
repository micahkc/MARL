import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Agent():
    def __init__(self, n):
        self.id = n

    class Actor(nn.Module):
        def __init__(self, hidden_dim=28):
            super().__init__()

            self.hidden = nn.Linear(28, hidden_dim)
            self.output = nn.Linear(hidden_dim, 2)

        def forward(self, s):
            outs = self.hidden(s)
            outs = F.relu(outs)
            logits = self.output(outs)
            return logits
    
    class Critic(nn.Module):
        def __init__(self, hidden_dim=16):
            super().__init__()

            self.hidden = nn.Linear(4, hidden_dim)
            self.output = nn.Linear(hidden_dim, 1)

        def forward(self, s):
            outs = self.hidden(s)
            outs = F.relu(outs)
            value = self.output(outs)
            return value
    
    def create_models(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = self.Actor().to(device)
        self.critic = self.Critic().to(device)

        self.optimizerA = optim.Adam(self.actor.parameters())
        self.optimizerC = optim.Adam(self.critic.parameters())

    def update_critic(self, observation, target):
        self.optimizerC.zero_grad()
        self.critic_loss.backward()
        self.optimizerC.step()

    def update_actor(self, observation, error, action):
        self.optimizerA.zero_grad()
        self.actor_loss.backward()
        self.optimizerA.step()
        
    