from unityagents import UnityEnvironment
import random
import numpy as np
import torch
from collections import deque
from datetime import datetime
from maddpg_agents import DDPGAgent, MADDPG

def main():
    env = UnityEnvironment(file_name='data/Tennis_Linux/Tennis.x86_64')
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # size of each action
    action_size = brain.vector_action_space_size
    # examine the state space 
    states = env_info.vector_observations
    obs_size = states.shape[1]
    state_size = obs_size * states.shape[0]

    agent_list = [
        DDPGAgent(obs_size, state_size, action_size, random_seed = 2018),
        DDPGAgent(obs_size, state_size, action_size, random_seed = 2019),
    ]
    agents = MADDPG(agent_list)

    # load trained model
    agents.maddpg_agent[0].actor_local.load_state_dict(torch.load('model/checkpoint_actor.pth'))
    agents.maddpg_agent[1].actor_local.load_state_dict(torch.load('model/checkpoint_actor.pth'))

    state = env.reset()
    env_info = env.reset(train_mode=False)[brain_name]
    obs_tuple = env_info.vector_observations
    for t in range(2000):
        action_tuple = agents.act(obs_tuple, add_noise=False)
        env_info = env.step(action_tuple)[brain_name]
        next_obs_tuple = env_info.vector_observations
        done_tuple = env_info.local_done
        obs_tuple = next_obs_tuple
        if np.any(done_tuple):
            break

    env.close()
    
if __name__ == "__main__":
    main()
