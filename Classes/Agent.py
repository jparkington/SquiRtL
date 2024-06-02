from collections         import deque
from DQN                 import DQN
from torch.nn.functional import smooth_l1_loss
from torch.optim         import Adam, lr_scheduler

import random
import torch

class Agent:
    def __init__(self, state_dim, action_space, save_dir, date, hyperparameters):
        self.action_space    = action_space
        self.curr_step       = 0
        self.date            = date
        self.device          = torch.device("cpu")
        self.dqn             = DQN(state_dim, len(action_space)).to(self.device)
        self.hyperparameters = hyperparameters
        self.memory          = deque(maxlen = hyperparameters.deque_size)
        self.optimizer       = Adam(self.dqn.parameters(), lr = hyperparameters.learning_rate)
        self.save_dir        = save_dir
        self.scheduler       = lr_scheduler.ExponentialLR(self.optimizer, gamma=hyperparameters.learning_rate_decay)
        self.state_dim       = state_dim
        self.target_dqn      = DQN(state_dim, len(action_space)).to(self.device)

        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()
    
    def act(self, state):
        if random.random() < self.hyperparameters.exploration_rate:
            action_idx = random.randint(0, len(self.action_space) - 1)
        else:
            state      = torch.tensor(state, dtype = torch.float32).unsqueeze(0).to(self.device)
            q_values   = self.dqn(state)
            action_idx = torch.argmax(q_values, dim = 1).item()

        self.hyperparameters.exploration_rate = max(
            self.hyperparameters.exploration_rate_min, 
            self.hyperparameters.exploration_rate * self.hyperparameters.exploration_rate_decay
        )
        
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        state, next_state = map(lambda x: torch.tensor(x, dtype = torch.float32).to(self.device), (state, next_state))
        action = torch.tensor([action], dtype = torch.long).to(self.device)
        reward = torch.tensor([reward], dtype = torch.float32).to(self.device)
        done   = torch.tensor([done],   dtype = torch.bool).to(self.device)
        
        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        batch = random.sample(self.memory, self.hyperparameters.batch_size)
        return map(torch.stack, zip(*batch))

    def learn(self):
        if self.curr_step < self.hyperparameters.burnin or self.curr_step % self.hyperparameters.learn_every != 0:
            return None, None
        
        if self.curr_step % self.hyperparameters.sync_every == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        if self.curr_step % self.hyperparameters.save_every == 0:
            self.save()
        
        state, next_state, action, reward, done = self.recall()
        q_values          = self.dqn(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values     = self.target_dqn(next_state).max(1)[0]
        expected_q_values = reward + self.hyperparameters.gamma * next_q_values * (1 - done.float())

        loss = smooth_l1_loss(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return q_values.mean().item(), loss.item()

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.dqn.load_state_dict(checkpoint["model"])
        self.hyperparameters.exploration_rate = checkpoint["exploration_rate"]
        print(f"Loading model at {path} with exploration rate {self.hyperparameters.exploration_rate}")

    def save(self):
        save_path = self.save_dir / f"dqn_net_{int(self.curr_step // self.hyperparameters.save_every)}.chkpt"
        torch.save({"model" : self.dqn.state_dict(), 
                    "exploration_rate" : self.hyperparameters.exploration_rate}, save_path)
        print(f"DQN Net saved to {save_path} at step {self.curr_step}")