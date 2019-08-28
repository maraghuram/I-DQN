# coding: utf-8

# ### Import required libraries

import argparse
from concurrent.futures import ThreadPoolExecutor

import torch, os, sys, random, gym, time
import numpy as np

from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from tqdm import tqdm
from collections import deque
from wrapper.gym_wrapper import make_atari, wrap_deepmind
from tensorboardX import SummaryWriter


def parse_args():
    # ### Specify Hyperparameters

    parser = argparse.ArgumentParser(description='double-dqn')

    parser.add_argument('--env', default='PongNoFrameskip-v4', help='gym environment (atari)')

    parser.add_argument('--max_timesteps', default=10000000)
    parser.add_argument('--max_episodes', default=10000)

    parser.add_argument('--max_exploration_timestep', default=1000000.0)
    parser.add_argument('--init_eps_value', default=1.0)
    parser.add_argument('--final_eps_value', default=0.01)

    parser.add_argument('--replay_size', default=10000)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--learning_rate', default=0.0001)
    parser.add_argument('--grad_clip', default=10.0)

    parser.add_argument('--train_freq', default=4)
    parser.add_argument('--print_freq', default=10000)
    parser.add_argument('--sync_freq', default=1000)
    parser.add_argument('--train_start', default=10000)
    parser.add_argument('--best_mean_100', default=15)

    return parser.parse_args()


# ### Global Variables


use_cuda = True
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


# ### Network Definition

class QNetwork(nn.Module):
    def __init__(self, input_shape, output_len, hidden_dim=256):
        super(QNetwork, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_len)
        )

        self.init_weights()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)   # NCHW format
        x = x.float()/255.0
        fx= self.conv(x).view(x.size()[0], -1)
        return self.fc(fx)

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)
                nn.init.constant_(layer.bias.data, 0.0)

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights.
        torch.save(self.state_dict(), suffix)

    def load_model_weights(self, model_file):
        self.load_state_dict(torch.load(model_file))


# class QNetwork(nn.Module):
#         def __init__(self, input_shape, output_len, hidden_dim=256):
#             super(QNetwork, self).__init__()
#
#             self.fc_net = nn.Sequential(
#                 nn.Linear(input_shape[0], 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 64),
#                 nn.ReLU(),
#                 nn.Linear(64, 64),
#                 nn.ReLU(),
#                 nn.Linear(64, output_len)
#             )
#
#         def forward(self, x):
#             # x = x.permute(0, 3, 1, 2)   # NCHW format
#             # x = x.float()/255.0
#             # fx = self.conv(x).view(x.size()[0], -1)
#             return self.fc_net(x)
#
#         def _get_conv_out(self, shape):
#             o = self.conv(Variable(torch.zeros(1, *shape)))
#             return int(np.prod(o.size()))
#
#         def save_model_weights(self, suffix):
#             # Helper function to save your model / weights.
#             torch.save(self.state_dict(), suffix)
#
#         def load_model_weights(self, model_file):
#             self.load_state_dict(torch.load(model_file))


# ### Helper Functions
# #### eps_greedy_action(), sample_memory()


def eps_greedy_action(curr_eps, q_values, n_action):
    if np.random.random() < curr_eps:
        action = np.random.choice(n_action)
    else:
        _, actions = torch.max(q_values, dim=0)
        action = int(actions.cpu().data.numpy())

    return action


replay_memory = []


def sample_memory(n_samples):
    return random.sample(replay_memory, n_samples)


def process_img(img):
    return np.array(img)
    # return np.array(img).reshape(4, 84, 84).astype(np.float32) / 255.0


def main(args, env):
    global replay_memory

    n_state = env.observation_space.shape
    n_action = env.action_space.n

    # ### Prepare for training

    state = env.reset()
    state_p = process_img(state).reshape(4, 84, 84).astype(np.float32) / 255.0

    cumulative_rewards_tracker = []
    loss_tracker = []
    cumulative_rewards = 0
    episode_length = 0

    net = QNetwork(state_p.shape, n_action)
    tgt = QNetwork(state_p.shape, n_action)
    # net = QNetwork(n_state, n_action)
    # tgt = QNetwork(n_state, n_action)
    if use_cuda:
        net = net.cuda()
        tgt = tgt.cuda()

    opt = Adam(net.parameters(), lr=args.learning_rate, weight_decay=0.0)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_directory = 'logs/%s/%s/%s/' % ('double-dqn-forexp', args.env, run_id)
    os.makedirs(log_directory) if not os.path.exists(log_directory) else None
    writer = SummaryWriter(comment="-" + run_id + "", log_dir=log_directory)
    best_mean_100 = args.best_mean_100  # Atleast positive reward

    with open('%s/run_args.txt' % log_directory, 'w') as f_out:
        f_out.write(str(args))

    loss = 0.0
    loss_tracker.append(loss)

    # Fill replay buffer
    replay_memory = deque(maxlen=args.replay_size)

    # ### Train

    for timesteps in tqdm(range(args.max_timesteps)):
        # Calculate exploration factor
        eps_value = max(args.final_eps_value,
                        args.init_eps_value - (
                        (args.init_eps_value - args.final_eps_value) * timesteps / args.max_exploration_timestep))

        with torch.no_grad():
        	q_values = net(Variable(FloatTensor(process_img(state)[None])))[0]
        act = eps_greedy_action(eps_value, q_values, n_action)

        next_state, reward, done, _ = env.step(act)
        reward_clipped = np.clip(reward, -1, 1)

        replay_memory.append((state, act, reward_clipped, next_state, 1.0 - float(done)))
        state = next_state
        cumulative_rewards += reward
        episode_length += 1

        if done:
            state = env.reset()
            cumulative_rewards_tracker.append(cumulative_rewards)

            curr_mean_100 = np.mean(cumulative_rewards_tracker[-100:])
            writer.add_scalar('reward_100', curr_mean_100, timesteps)
            writer.add_scalar('loss_100', np.mean(loss_tracker[-100:]), timesteps)
            writer.add_scalar('reward', cumulative_rewards_tracker[-1], timesteps)
            writer.add_scalar('loss', loss_tracker[-1], timesteps)
            writer.add_scalar('length', episode_length, timesteps)
            writer.add_scalar('epsilon', eps_value, timesteps)

            if curr_mean_100 > best_mean_100:
                best_mean_100 = curr_mean_100
                save_path = '%s/best-models/model.tar' % (log_directory)
                os.makedirs('%s/best-models' % log_directory) if not os.path.exists(
                    '%s/best-models' % log_directory) else None
                net.save_model_weights(save_path)

                # mean_rewards_test = test(net, args)
                # print('Avg. rewards(test): %.3f' % mean_rewards_test)
                # writer.add_scalar('mean_rewards_test', mean_rewards_test, timesteps)

            cumulative_rewards = 0
            episode_length = 0

        if timesteps < args.train_start:
            continue

        if timesteps % args.train_freq == 0:
            # Sample batch
            samples = sample_memory(args.batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []

            for a, b, c, d, e in samples:
                state_batch.append(ByteTensor(process_img(a)))
                action_batch.append(b)
                reward_batch.append(c)
                next_state_batch.append(ByteTensor(process_img(d)))
                done_batch.append(e)

            reward_batch = np.array(reward_batch)
            done_batch = np.array(done_batch)
            action_batch = np.array(action_batch, dtype=np.int64)


            opt.zero_grad()

            # Convert to Variable
            state_batch = Variable(torch.stack(state_batch))
            action_batch = Variable(LongTensor(action_batch))
            reward_batch = Variable(FloatTensor(reward_batch))
            done_batch = Variable(FloatTensor(done_batch))

            # Calculate loss

            state_qvalues = net(state_batch)
            curr_qvalues = state_qvalues[range(args.batch_size), action_batch.data]

            # with torch.no_grad():
            next_state_batch = Variable(torch.stack(next_state_batch))
            next_state_qvalues = net(next_state_batch)
            next_state_qvalues = Variable(next_state_qvalues.data)

            _, next_state_best_action = torch.max(next_state_qvalues, 1)

            target_qvalues = tgt(next_state_batch)
            target_qvalues = target_qvalues[range(args.batch_size), next_state_best_action.data]

            # I thought this step doesn't matter much
            # Turns out, this step alone can drive training to death
            # The goal/end/final state never has any supervised targets and hence its initial predictions are random
            # If the value for this state is not set to zero (or some constant), the network tries to play
            # catch-up to
            # its own random predictions and so the losses spiral out of control (the loss is meaningless)

            target_qvalues = done_batch * target_qvalues

            # q(s, a) = r(s, a, s') + Y max{q(s', a') over a'}
            target_qvalues = (args.gamma * target_qvalues) + reward_batch
            target_qvalues = target_qvalues.detach()

            # loss = 0.5 * (curr_qvalues - target_qvalues).pow(2).sum()
            # loss = loss / args.batch_size
            loss = nn.SmoothL1Loss()(curr_qvalues, target_qvalues)

            # Backward
            loss.backward()

            # Clip gradients
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    torch.nn.utils.clip_grad_norm_(layer.parameters(), args.grad_clip)

            opt.step()

            loss_tracker.append(loss.item())

        if timesteps % args.print_freq == 0:
            print('Avg. loss: %.6f, Avg. cumulative reward: %.3f' % (np.mean(loss_tracker[-100:]),
                                                                     np.mean(cumulative_rewards_tracker[-100:])))
            print('Episodes completed: %d' % len(cumulative_rewards_tracker))

        if timesteps % args.sync_freq == 0:
            tgt.load_state_dict(net.state_dict())

    writer.close()


def test_one_episode(env, model, test_eps=0.001):
    state = env.reset()
    done = False
    tot_rewards = 0

    while not done:
        q_values = model(Variable(ByteTensor(process_img(state)[None]), volatile=True))
        q_values = q_values[0]
        act = eps_greedy_action(test_eps, q_values, env.action_space.n)

        next_state, reward, done, _ = env.step(act)
        state = next_state
        tot_rewards += reward

    return tot_rewards


def test(model, args):
    mean_rewards_test = []

    with ThreadPoolExecutor(max_workers=20) as executor:
        thread_handlers = []
        for i in range(50):
            env = make_atari(args.env)
            env = wrap_deepmind(env, frame_stack=True, scale=False, clip_rewards=False, episode_life=False)
            thread_handlers.append(executor.submit(test_one_episode, env, model))

        for thread in thread_handlers:
            mean_rewards_test.append(thread.result())

    return np.mean(mean_rewards_test)


if __name__ == '__main__':
    # ### Setup Gym environment
    atari_args = parse_args()

    atari_env = make_atari(atari_args.env)
    atari_env = wrap_deepmind(atari_env, frame_stack=True, scale=False, clip_rewards=False, episode_life=False)
    # atari_env = gym.make('CartPole-v0')

    main(atari_args, atari_env)

    atari_env.close()
