from utils import cross_loss_curve, GAMA_connect,send_to_GAMA,reset
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
torch.set_default_tensor_type(torch.DoubleTensor)

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.states_next = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.states_next[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 64)
        self.linear2 = nn.Linear(64, 128)
        self.LSTM_layer_3 = nn.LSTM(128,64,1, batch_first=True)
        self.linear4 = nn.Linear(64,32)
        self.mu = nn.Linear(32,self.action_size)  #256 linear2
        self.sigma = nn.Linear(32,self.action_size)

    def forward(self, state):
        output_1 = F.relu(self.linear1(state))
        output_2 = F.relu(self.linear2(output_1))
        #LSTM
        output_2  = output_2.unsqueeze(0)
        output_3 , output = self.LSTM_layer_3(output_2)
        a,b,c = output_3.shape
        #
        output_4 = F.relu(self.linear4(output_3.view(-1,c))) #
        mu = 2 * torch.tanh(self.mu(output_4))   #有正有负 sigmoid 0-1
        sigma = F.relu(self.sigma(output_4)) + 0.001 
        mu = torch.diag_embed(mu).to(device)
        sigma = torch.diag_embed(sigma).to(device)  # change to 2D
        dist = MultivariateNormal(mu,sigma)  #N(μ，σ^2)
        entropy = dist.entropy().mean()
        action = dist.sample()
        action_logprob = dist.log_prob(action)     
        return action,action_logprob,entropy

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 64)
        self.linear2 = nn.Linear(64, 128)
        self.LSTM_layer_3 = nn.LSTM(128,64,1, batch_first=True)
        self.linear4 = nn.Linear(64,32) #
        self.linear5 = nn.Linear(32, action_size)

    def forward(self, state):
        output_1 = F.relu(self.linear1(state))
        output_2 = F.relu(self.linear2(output_1))
        #LSTM
        output_2  = output_2.unsqueeze(0)
        output_3 , output = self.LSTM_layer_3(output_2)
        a,b,c = output_3.shape
        #
        output_4 = F.relu(self.linear4(output_3.view(-1,c))) 
        value  = torch.tanh(self.linear5(output_4))
        return value#,output

class PPO:
    def __init__(self, state_dim, action_dim, lr,gamma, K_epochs, eps_clip):
        self.state_dim = state_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.max_grad_norm = 0.5
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.A_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=(0.95, 0.999)) #更新新网
        self.C_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(0.95, 0.999)) #更新新网

        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        memory.states.append(state)
        action,action_logprob,entropy = self.actor(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        action.detach()
        return torch.clamp(action, -0.5, 0.8) 
    
    def update(self, memory,lr,advantages,done,loss):
        # 更新lr
        self.A_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=(0.95, 0.999)) #更新新网
        self.C_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(0.95, 0.999)) #更新新网
        # Monte Carlo estimate of rewards: MC
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal == 1:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards).to(device)
        
        # convert list to tensor  截断所有旧网络出来的值，旧网络计算只会用到 logP
        old_states =torch.cat(memory.states).to(device).detach()
        old_states_next = torch.cat(memory.states_next).to(device).detach()  
        old_actions = torch.cat(memory.actions).to(device).detach()
        old_logprobs = torch.cat(memory.logprobs).to(device).detach() 
        print("old_states_last",old_states[old_states.size()[0]-1],"rewards_last",rewards[old_states.size()[0]-1])

        for _ in range(self.K_epochs):
            for i in range(old_states.size()[0]):
                # Evaluating old actions and values :
                state_values= self.critic(old_states[i].reshape(1,self.state_dim))
                # Surrogate Loss: # TD:r(s) + v(s+1) - v(s)  # MC = R-V(s)
                advantages = rewards[i].detach() - state_values  #MC use
                c_loss = F.smooth_l1_loss(rewards[i].reshape(1,1).detach(), state_values)
                
                action,logprobs,entropy = self.actor(old_states[i].reshape(1,self.state_dim)) 
                ratios = torch.exp(logprobs - old_logprobs[i].detach()) #log转正数probability

                # ratio (ppi_theta/ppi_theta__old):
                surr1 = ratios * advantages.detach()
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages.detach()
                a_loss = -torch.min(surr1, surr2)  #+ 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy #均方损失函数
                
                self.A_optimizer.zero_grad()
                self.C_optimizer.zero_grad()
                a_loss.backward() 
                #nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                c_loss.backward()
                #nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.A_optimizer.step()
                self.C_optimizer.step()
                loss.append(c_loss)

        return loss

def main():
    ############## Hyperparameters ##############
    K_epochs = 3          # update policy for K epochs  lr太大会出现NAN?
    eps_clip = 0.2            
    gamma = 0.9 # 要较弱；较强关联？ 对每一正确步也有打击       
    
    lr_first = 0.00001               
    lr = lr_first   #random_seed = None
    state_dim = 6
    action_dim = 1 
    #(self, state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip)
    actor_path = os.getcwd()+'\\GAMA_python\\PPO_Navigation_Model\\weight\\ppo_MC_actor.pkl'
    critic_path = os.getcwd()+'\\GAMA_python\\PPO_Navigation_Model\\weight\\ppo_MC_critic.pkl'
    ################ load ###################

    ppo =  PPO(state_dim, action_dim, lr, gamma, K_epochs, eps_clip)
    if os.path.exists(actor_path):
        ppo.actor.load_state_dict(torch.load(actor_path ))
        print('Actor Model loaded')
    if os.path.exists(critic_path):
        ppo.critic.load_state_dict(torch.load(critic_path))
        print('Critic Model loaded')
    print("Waiting for GAMA...")

    ################### initialization ########################
    save_curve_pic = os.getcwd()+'\\GAMA_python\\PPO_Navigation_Model\\result\\PPO_MC_loss_curve.png'
    save_critic_loss = os.getcwd()+'\\GAMA_python\\PPO_Navigation_Model\\training_data\\PPO_MC_critic_loss.csv'
    save_reward = os.getcwd()+'\\GAMA_python\\PPO_Navigation_Model\\training_data\\PPO_MC_reward.csv'
    reset()
    memory = Memory()

    episode = 339
    advantages = 0 #global value
    loss = []
    total_loss = []
    rewards = []
    total_rewards = []
    test = "GAMA"
    state,reward,done,time_pass,over = GAMA_connect(test) #connect
    #[real_speed/10, target_speed/10, elapsed_time_ratio, distance_left/100,distance_front_car/10,distance_behind_car/10,reward,done,over]
    print("done:",done,"timepass:",time_pass)

    ##################  start  #########################
    while over!= 1:
        #普通の場合
        if(done == 0 and time_pass != 0):  
            rewards.append(reward)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            state = torch.DoubleTensor(state).reshape(1,6).to(device) 
            memory.states_next.append(state)

            action = ppo.select_action(state, memory)
            send_to_GAMA([[1,float(action.cpu().numpy()*10)]])

        # 終わり 
        elif done == 1:
            #先传后计算
            send_to_GAMA( [[1,0]] ) 
            rewards.append(reward) 

            state = torch.DoubleTensor(state).reshape(1,6).to(device) #转化成1行
            memory.states_next.append(state)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            loss = ppo.update(memory,lr,advantages,done,loss)
            memory.clear_memory()

            print("----------------------------------Net_Trained---------------------------------------")
            print('--------------------------Iteration:',episode,'over--------------------------------')
            episode += 1
            loss_sum = sum(loss).cpu().detach().numpy()
            total_loss.append(loss_sum)
            total_reward = sum(rewards)
            total_rewards.append(total_reward)
            cross_loss_curve(loss_sum.squeeze(0),total_reward,save_curve_pic,save_critic_loss,save_reward)
            rewards = []
            loss = []
            if episode >30  : #50 100
                lr = lr_first * (0.94 ** ((episode-20) // 10))
                #if episode > 80:
                 #   lr_first = 0.0001
                  #  lr = lr_first * (0.94 ** ((episode-70) // 10))
            torch.save(ppo.actor.state_dict(),actor_path)
            torch.save(ppo.critic.state_dict(),critic_path)

        #最初の時
        else:
            print('Iteration:',episode)
            state = torch.DoubleTensor(state).reshape(1,6).to(device) 
            action = ppo.select_action(state, memory)
            print("acceleration: ",action) #.cpu().numpy()
            send_to_GAMA([[1,float(action.cpu().numpy()*10)]])

        state,reward,done,time_pass,over = GAMA_connect(test)
    return None 

if __name__ == '__main__':
    main()
