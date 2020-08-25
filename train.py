"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
import torch.nn as nn
from src.env import MultipleEnvironments
from src.model import PPO
from src.model import RandomNetwork
from src.process import eval
import torch.multiprocessing as _mp
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import shutil

import wandb
wandb.init(project="procgen-super-mario-ppo", reinit=True)


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Super Mario Bros""")
    parser.add_argument("--start_level", type=int, default=0)
    parser.add_argument("--num_levels", type=int, default=0)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/ppo_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--channels", type=int, default=3, help="image channels, rgb or grey")
    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    mp = _mp.get_context("spawn")
    envs = MultipleEnvironments(opt.start_level, opt.num_levels, opt.action_type, opt.num_processes)
    model = PPO(envs.num_states*opt.channels, envs.num_actions)
    if torch.cuda.is_available():
        model.cuda()
    model.share_memory()
    process = mp.Process(target=eval, args=(opt, model, envs.num_states, envs.num_actions))
    process.start()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
    if torch.cuda.is_available():
        curr_states = curr_states.cuda()

    # randon network init
    random_model = RandomNetwork(envs.num_states*opt.channels)
    if torch.cuda.is_available():
        random_model.cuda()
    mse_loss = nn.MSELoss()

    curr_episode = 0
    total_steps = 0
    while True:
        # if curr_episode % opt.save_interval == 0 and curr_episode > 0:
        #     torch.save(model.state_dict(),
        #                "{}/ppo_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
        #     torch.save(model.state_dict(),
        #                "{}/ppo_super_mario_bros_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage, curr_episode))
        curr_episode += 1
        old_log_policies = []
        actions = []
        values = []
        states = []
        states_raw = []
        rewards = []
        dones = []
        rewards_log = []
        reward_ep = np.zeros(opt.num_processes)
        
        # randon network re-init
        random_model.re_init()
        if torch.cuda.is_available():
            random_model.cuda()

        for _ in range(opt.num_local_steps):
            states_raw.append(curr_states)
            # random net
            with torch.no_grad():
                curr_states_random = random_model(curr_states)
            # curr_states_random = curr_states_random.cuda()
            states.append(curr_states_random)
            #
            # curr_states_random.to('cpu')
            # del curr_states_random
            # torch.cuda.empty_cache()
            logits, value, penultimate_random = model(curr_states_random)
            values.append(value.squeeze())
            policy = F.softmax(logits, dim=1)
            old_m = Categorical(policy)
            action = old_m.sample()
            actions.append(action)
            old_log_policy = old_m.log_prob(action)
            old_log_policies.append(old_log_policy)
            if torch.cuda.is_available():
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
            else:
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]

            state, reward, done, info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
            
            for i in range(opt.num_processes):
                reward_ep[i] += reward_ep[i] + reward[i]

            for j in range(len(done)):
                if done[j]:
                    rewards_log.append(reward_ep[j])
                    reward_ep[j] = 0

            state = torch.from_numpy(np.concatenate(state, 0))
            if torch.cuda.is_available():
                state = state.cuda()
                reward = torch.cuda.FloatTensor(reward)
                done = torch.cuda.FloatTensor(done)
            else:
                reward = torch.FloatTensor(reward)
                done = torch.FloatTensor(done)
            rewards.append(reward)
            dones.append(done)
            curr_states = state

            total_steps += len(rewards)*opt.num_processes

        with torch.no_grad():
            curr_states_random = random_model(curr_states)
        # curr_states_random = curr_states_random.cuda()
        _, next_value, _ = model(curr_states_random)
        #
        # curr_states_random.to('cpu')
        # del curr_states_random
        # torch.cuda.empty_cache()
        #
        next_value = next_value.squeeze()
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)
        states_raw = torch.cat(states_raw)
        gae = 0
        R = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() * (1 - done) - value.detach()
            next_value = value
            R.append(gae + value)
        R = R[::-1]
        R = torch.cat(R).detach()
        advantages = R - values
        for i in range(opt.num_epochs):
            indice = torch.randperm(opt.num_local_steps * opt.num_processes)
            for j in range(opt.batch_size):
                batch_indices = indice[
                                int(j * (opt.num_local_steps * opt.num_processes / opt.batch_size)): int((j + 1) * (
                                        opt.num_local_steps * opt.num_processes / opt.batch_size))]
                logits, value, penultimate_random = model(states[batch_indices])
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(actions[batch_indices])
                ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
                actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices],
                                                   torch.clamp(ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon) *
                                                   advantages[
                                                       batch_indices]))
                # critic_loss = torch.mean((R[batch_indices] - value) ** 2) / 2
                critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                entropy_loss = torch.mean(new_m.entropy())
                # random net loss
                _, _, penultimate_raw = model(states_raw[batch_indices])
                random_net_loss = mse_loss(penultimate_random, penultimate_raw)

                total_loss = actor_loss + critic_loss - opt.beta * entropy_loss + random_net_loss
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        print("Episode: {}. Total loss: {}".format(curr_episode, total_loss))
        wandb.log({"total loss": total_loss,
                    "ep reward": np.mean(rewards_log),
                    "global steps": total_steps,})

if __name__ == "__main__":
    opt = get_args()
    train(opt)
