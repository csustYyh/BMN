import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import random
import numpy as np
import math
# only for trainning
from model.repaly_buffer import ReplayBuffer
import torch.optim as optim



# policy_net 和 target_net 的网络结构
# input:[it.key, memory.key] (1, 64+64, H/16, W/16)
# output:[1, Q1&Q2] (1, 2) 
class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()
        self.conv1 = nn.Conv2d(64 + 64, 256, 1, 1, 0) # (B, 256, H/16, W/16)
        self.conv2 = nn.Conv2d(256, 512, 3, 1, 1) # (B, 512, H/16, W/16)
        self.conv3 = nn.Conv2d(512, 1024, 3, 2, 1) # (B, 1024, H/32, W/32)
        self.conv4 = nn.Conv2d(1024, 1024, 3, 2, 1) # (B, 1024, H/64, W/64)
        self.avp = nn.AdaptiveAvgPool2d(1) # (B, 1024, 1, 1)
        self.fc1 = torch.nn.Linear(1024, 1024)
        self.fc2 = torch.nn.Linear(1024, 2)

        for i in [self.conv1, self.conv2, self.conv3, self.conv4]:
            nn.init.kaiming_normal_(i.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(i.bias, 0)

    def forward(self, state):  # (B, 64 + 64, H/16, W/16)
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.avp(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1) # (B, 1024)
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)

        return x 

class Agent(nn.Module):
    def __init__(self, device):
        super(Agent, self).__init__()
        self.device = device

        self.EPS_START = 0.7
        self.EPS_END = 0.25
        self.EPS_DECAY = 500
        # 执行动作的次数/训练次数
        self.steps_done = 0
        # 动作空间大小
        self.action_size = 2
        # target网络更新率
        self.update_rate = 0.05
        # 经验池
        self.memory_size = 100000
        
        self.GAMMA = 0.95
        self.policy_net = Brain()
        self.target_net = Brain()

        self.target_net.load_state_dict(self.policy_net.state_dict())

        params = self.policy_net.parameters()

        self.optimizer = optim.Adam(params, lr=0.000005, weight_decay=0.0005)
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

    def set_train(self):
        self.policy_net.train()
        self.target_net.train()

    def set_eval(self):
        self.policy_net.eval()
        self.target_net.eval()


    # 采取动作
    def rl_action(self, state, train, verbose=False):
        
        if train == True:
            with torch.no_grad():
                Q_Value = self.policy_net(state)
            action = Q_Value.argmax()
            return action

        self.steps_done += 1

        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-0.5 * self.steps_done / self.EPS_DECAY)

        rand_flag = random.random()
        if rand_flag > eps_threshold:
            if verbose:

                print(f"step:{self.steps_done}, rand_flag:{rand_flag:.4f}, eps_threshold:{eps_threshold:.4f}, "
                      f"frame index was selected by agent")
            with torch.no_grad():
                self.policy_net.eval()
                Q_Value = self.policy_net(state)
                self.policy_net.train()
            action = Q_Value.argmax()
            return action
        else:
            if verbose:
                print(f"step:{self.steps_done}, rand_flag:{rand_flag:.4f}, eps_threshold:{eps_threshold:.4f}, "
                      f"frame index was selected randomly")
            action_idx = np.array(range(self.action_size))
            action = random.choice(action_idx)
            return action

    # 更新智能体
    def rl_update(self, transition_dict, count):
        torch.autograd.set_grad_enabled(True) 
        if transition_dict is None:
            print('no input')
            return
        batch_size = 4

        states = transition_dict['states']
        actions = transition_dict['actions']
        rewards = transition_dict['rewards']
        next_states = transition_dict['next_states']
        done = transition_dict['done']
        
        # 转换数据为张量
        done = done.view(-1, 1).float()
        rewards = rewards.view(-1, 1)
        rewards = rewards.to(self.device)
        done = done.to(self.device)
        actions = actions.view(-1, 1)
        actions = actions.to(self.device)

        with torch.no_grad():
            self.set_eval()
            output = self.policy_net(next_states)
            next_action = output.max(1)[1].view(batch_size, -1)

            Q_next_state_target = self.target_net(next_states)
            Q_next_state = Q_next_state_target.gather(1, next_action.long()).detach()

            Q_state_action_target = rewards + (1 - done) * self.GAMMA * Q_next_state

        Q_state = self.policy_net(states)
        Q_state_action = Q_state.gather(1, actions.long())
        # ====== update policy net ======
        loss= torch.mean(F.mse_loss(Q_state_action, Q_state_action_target)).requires_grad_(True)
        self.optimizer.zero_grad()
        loss.backward()  

        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # ====== update target net ======
        if np.random.random() < self.update_rate:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if count >= 10000 and count % 10000 == 0:
            torch.save(self.policy_net.state_dict(), os.path.join('./agent_weights', 'rl_{}.pth'.format(count)))
        torch.autograd.set_grad_enabled(False) 
        return loss.item()






