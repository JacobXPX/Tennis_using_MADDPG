from unityagents import UnityEnvironment
import random
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
def MSG(txt):
    print('\n',datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), str(txt))
from maddpg_agents import DDPGAgent, MADDPG

def maddpg(env, agents, n_episodes=3500, max_t=2000, print_every=100):
    MSG('start!')
    brain_name = env.brain_names[0]
    scores_deque = deque(maxlen=print_every)
    scores_idx_deque = deque(maxlen=print_every)
    scores = []
    best_score = 0.
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        obs_tuple = env_info.vector_observations
        agents.reset()
        agent_scores = np.zeros(len(agents.maddpg_agent))
        for t in range(max_t):
            action_tuple = agents.act(obs_tuple)                    # select an action (for each agent)
            env_info = env.step(action_tuple)[brain_name]           # send all actions to tne environment
            next_obs_tuple = env_info.vector_observations           # get next state (for each agent)
            reward_tuple = env_info.rewards                         # get reward (for each agent)
            done_tuple = env_info.local_done                        # see if episode finished
            agent_scores += reward_tuple
            agents.step(obs_tuple, action_tuple, reward_tuple, next_obs_tuple, done_tuple)
            obs_tuple = next_obs_tuple                              # roll over states to next time step
            if np.any(done_tuple):                                  # exit loop if episode finished
                break
        score, idx = np.max(agent_scores), np.argmax(agent_scores)
        scores_deque.append(score)
        scores_idx_deque.append(idx)
        scores.append(score)
        if score > best_score:
            torch.save(agents.maddpg_agent[idx].actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agents.maddpg_agent[idx].critic_local.state_dict(), 'checkpoint_critic.pth')
            best_score = score
        if i_episode % (print_every) == 0:
            print('\rEpisode {}\tWinner is agent {}\tAverage Score on 100 Episode: {:.3f}'.format(i_episode, int(round(np.mean(scores_idx_deque))), np.mean(scores_deque)))
    MSG('end!') 
    return scores

def main():
    env = UnityEnvironment(file_name="data/Tennis_Linux/Tennis.x86_64")
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
    scores = maddpg(env, agents)
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    fig.savefig('score.png')
    plt.close(fig)
if __name__ == "__main__":
    main()
    
    
