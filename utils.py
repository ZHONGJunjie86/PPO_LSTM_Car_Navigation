import matplotlib.pyplot as plt
import numpy as np
import time,random
import os 

from_GAMA_1 = os.getcwd()+'\\GAMA_python\\PPO_Navigation_Model\\GAMA_R\\GAMA_intersection_data_1.csv'
from_GAMA_2 = os.getcwd()+'\\GAMA_python\\PPO_Navigation_Model\\GAMA_R\\GAMA_intersection_data_2.csv'
from_python_1 = os.getcwd()+'\\GAMA_python\\PPO_Navigation_Model\\GAMA_R\\python_AC_1.csv'
from_python_2 = os.getcwd()+'\\GAMA_python\\PPO_Navigation_Model\\GAMA_R\\python_AC_2.csv'


def reset():
    f=open(from_GAMA_1, "r+")
    f.truncate()
    f=open(from_GAMA_2, "r+")
    f.truncate()
    f=open(from_python_1, "r+")
    f.truncate()
    f=open(from_python_2, "r+")
    f.truncate()
    return_ = [0]
    np.savetxt(from_python_1,return_,delimiter=',')
    np.savetxt(from_python_2,return_,delimiter=',')

def cross_loss_curve(critic_loss,total_rewards,save_curve_pic,save_critic_loss,save_reward):
    critic_loss = np.hstack((np.loadtxt(save_critic_loss, delimiter=","),critic_loss))
    reward = np.hstack((np.loadtxt(save_reward, delimiter=",") ,total_rewards))
    plt.plot(np.array(critic_loss), c='b', label='critic_loss')
    plt.plot(np.array(reward), c='r', label='total_rewards')
    plt.legend(loc='best')
    #plt.ylim(-15,15)
    plt.ylim(-0.2,0.15)
    plt.ylabel('critic_loss') 
    plt.xlabel('training steps')
    plt.grid()
    plt.savefig(save_curve_pic)
    plt.close()
    np.savetxt(save_critic_loss,critic_loss,delimiter=',')
    np.savetxt(save_reward,reward,delimiter=',')

def send_to_GAMA(to_GAMA):
    error = True
    while error == True:
        try:
            np.savetxt(from_python_1,to_GAMA,delimiter=',')
            np.savetxt(from_python_2,to_GAMA,delimiter=',')
            error = False
        except(IndexError,FileNotFoundError,ValueError,OSError,PermissionError):  
            error = True 

#[real_speed/10, target_speed/10, elapsed_time_ratio, distance_left/100,distance_front_car/10,distance_behind_car/10,reward,done,over]
def GAMA_connect(test):
    error = True
    while error == True:
        try:
            time.sleep(0.003)
            if(random.random()>0.3):
                state = np.loadtxt(from_GAMA_1, delimiter=",")
            else:
                state = np.loadtxt(from_GAMA_2, delimiter=",")
            time_pass = state[2]
            error = False
        except (IndexError,FileNotFoundError,ValueError,OSError):
            time.sleep(0.003)
            error = True
        
    reward = state[6]
    done = state[7]  # time_pass = state[6]
    over = state [8] 
    #print("Recived:",state," done:",done)
    state = np.delete(state, [6,7,8], axis = 0) #4,5,
    error = True
    while error == True:
        try:
            f1=open(from_GAMA_1, "r+")
            f1.truncate()
            f2=open(from_GAMA_2, "r+")
            f2.truncate()
            error = False
        except (IndexError,FileNotFoundError,ValueError,OSError):
            time.sleep(0.003)
            error = True

    return state,reward,done,time_pass,over,