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

    parser.add_argument('--max_timesteps', default=30000000)
    parser.add_argument('--max_episodes', default=10000)

    parser.add_argument('--max_exploration_timestep', default=1000000.0)
    parser.add_argument('--init_eps_value', default=1.0)
    parser.add_argument('--final_eps_value', default=0.01)

    parser.add_argument('--replay_size', default=10000)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--learning_rate', default=0.00025)
    parser.add_argument('--min_lr', default=0.000001)
    parser.add_argument('--grad_clip', default=10.0)

    parser.add_argument('--train_freq', default=4)
    parser.add_argument('--print_freq', default=10000)
    parser.add_argument('--sync_freq', default=1000)
    # parser.add_argument('--test_freq', default=50000)
    parser.add_argument('--train_start', default=10000)
    parser.add_argument('--best_mean_100', default=15)

    parser.add_argument('--hidden_dim', default=256)
    parser.add_argument('--exploration_factor', default=0.001)
    parser.add_argument('--l1', default=1.0)
    parser.add_argument('--l2', default=1.0)
    parser.add_argument('--l3', default=0.1)
    parser.add_argument('--l4', default=0.01)

    return parser.parse_args()


# ### Global Variables
# lambda_reg = 0.001
n_aggregates = 20
min_val = -25
max_val = 25
agg_values = np.linspace(min_val, max_val, n_aggregates)

use_cuda = True
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

mean_pixel_image = Variable(FloatTensor(np.zeros((84, 84, 4))), requires_grad=False)

# ### Network Definition

class QNetwork(nn.Module):
    def __init__(self, input_shape, output_len, hidden_dim):
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

        self.fc = nn.Linear(conv_out_size, hidden_dim)

        self.deconv_fc = nn.Linear(hidden_dim, conv_out_size)

        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_shape[0], kernel_size=8, stride=4)
        )

        # self.query_embed = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.key_matrix = nn.Parameter(torch.randn(output_len, n_aggregates, hidden_dim))
        self.register_buffer('agg_values', Variable(FloatTensor(agg_values).view(n_aggregates, 1)))

        self.init_weights()

    def forward(self, x):
        x = x.float() - mean_pixel_image
        x = x / 255.0
        x = x.permute(0, 3, 1, 2)   # NCHW format

        fx= self.conv(x).view(x.size()[0], -1)

        x_embed = self.fc(fx)

        # q = torch.mm(x_embed, torch.t(self.query_embed))
        q = x_embed

        att_keys = torch.matmul(self.key_matrix, torch.t(q))
        soft_keys = nn.Softmax(dim=1)(att_keys)

        att_values = soft_keys * self.agg_values
        att_values = att_values.sum(dim=1)

        # att_log_px = soft_keys.log()
        # att_entropy = -soft_keys * att_log_px
        # att_loss = torch.sum(att_entropy)

        reconstructed_img = self.deconv(self.deconv_fc(x_embed).view(x.size()[0], 64, 7, 7))
        reconstruction_loss = (reconstructed_img - x).pow(2).sum()

        return torch.t(att_values), soft_keys, reconstruction_loss

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform(layer.weight.data)
                nn.init.constant(layer.bias.data, 0.0)

            if isinstance(layer, nn.Linear):
                nn.init.normal(layer.weight.data, 0, 0.1)

            if isinstance(layer, nn.Parameter):
                nn.init.normal(layer.data, 0, 0.1)

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights.
        torch.save(self.state_dict(), suffix)

    def load_model_weights(self, model_file):
        self.load_state_dict(torch.load(model_file))


# ### Helper Functions
# #### eps_greedy_action(), sample_memory()

def action_selector(q_distributions, lambda_exp):
    means = torch.mm(q_distributions.data, FloatTensor(agg_values).view(n_aggregates, 1))
    variances = torch.mm(q_distributions.data, FloatTensor(agg_values).view(n_aggregates, 1).pow(2))
    variances = variances - means.pow(2)
    std_devs = variances.sqrt()

    optimistic_returns = means + lambda_exp * std_devs
    _, action = torch.max(optimistic_returns, dim=0)
    action = int(action)

    return action


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
    # return np.array(img).astype(np.float32) - mean_pixel_image
    # return np.array(img).reshape(4, 84, 84).astype(np.float32) / 255.0


def main(args, env):
    global replay_memory, mean_pixel_image

    n_state = env.observation_space.shape
    n_action = env.action_space.n

    # ### Prepare for training

    state = env.reset()
    state_p = np.array(state).reshape(4, 84, 84).astype(np.float32) / 255.0

    cumulative_rewards_tracker = []
    loss_tracker = []
    cumulative_rewards = 0
    episode_length = 0

    net = QNetwork(state_p.shape, n_action, args.hidden_dim)
    tgt = QNetwork(state_p.shape, n_action, args.hidden_dim)
    # net = QNetwork(n_state, n_action)
    # tgt = QNetwork(n_state, n_action)
    if use_cuda:
        net = net.cuda()
        tgt = tgt.cuda()

    opt = Adam(net.parameters(), lr=args.learning_rate, weight_decay=0.0)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_directory = 'logs/%s/%s/%s/' % ('ucb-new-experiments', args.env, run_id)
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

        # with torch.no_grad():
        q_values, q_dist, _ = net(Variable(ByteTensor(process_img(state)[None]), volatile=True))
        q_values = q_values[0]
        q_dist = q_dist[:, :, 0]
        act = action_selector(q_dist, args.exploration_factor)

        next_state, reward, done, _ = env.step(act)

        cumulative_rewards += reward
        replay_memory.append((state, act, reward, next_state, 1.0 - float(done)))
        state = next_state

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
            mean_pixel_image = mean_pixel_image + Variable(FloatTensor(process_img(state).astype(np.float32)/args.train_start), requires_grad=False)
            # mean_pixel_image = mean_pixel_image + (np.array(state).astype(np.float32)/args.train_start)
            # np.save(log_directory + "/mean_pixel_image.npy", mean_pixel_image)

            continue

        if timesteps == args.train_start:
            np.save(log_directory + "/mean_pixel_image.npy", mean_pixel_image.cpu().data.numpy())

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

            state_qvalues, soft_att_keys, re_loss = net(state_batch)
            curr_qvalues = state_qvalues[range(args.batch_size), action_batch.data]

            # with torch.no_grad():
            next_state_batch = Variable(torch.stack(next_state_batch))
            next_state_qvalues, _, _ = net(next_state_batch)
            next_state_qvalues = Variable(next_state_qvalues.data)

            _, next_state_best_action = torch.max(next_state_qvalues, 1)

            target_qvalues, target_soft_att_keys, _ = tgt(next_state_batch)
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

            # Calculate diversity loss
            eye = Variable(torch.eye(args.batch_size))
            if use_cuda:
                eye = eye.cuda()
            soft_keys_reshape = soft_att_keys.view(-1, args.batch_size)
            diversity_penalty = torch.norm(torch.mm(torch.t(soft_keys_reshape), soft_keys_reshape) - eye).pow(2)/args.batch_size

            # Calculate distributional loss
            atoms_next = reward_batch.view(-1, 1) + args.gamma * (done_batch.view(-1, 1)) * tgt.agg_values.view(1, -1)
            atoms_next = torch.clamp(atoms_next, min_val, max_val)
            delta_atom = (max_val - min_val) / float(n_aggregates - 1)\

            b = (atoms_next - min_val) / delta_atom
            l = b.floor()
            u = b.ceil()

            idx_actions = next_state_best_action.view(1, 1, -1).expand(-1, n_aggregates, -1)
            target_soft_att_keys = torch.t(target_soft_att_keys.gather(0, idx_actions).squeeze(0))

            d_m_l = (u + (l == u).float() - b) * target_soft_att_keys
            d_m_u = (b - l) * target_soft_att_keys

            target_prob = FloatTensor(args.batch_size, n_aggregates).fill_(0)
            for i in range(args.batch_size):
                target_prob[i].index_add_(0, l[i].long().data, d_m_l[i].data)
                target_prob[i].index_add_(0, u[i].long().data, d_m_u[i].data)

            curr_idx_actions = action_batch.view(1, 1, -1).expand(-1, n_aggregates, -1)
            soft_att_keys = torch.t(soft_att_keys.gather(0, curr_idx_actions).squeeze(0))

            soft_att_keys = soft_att_keys + 1e-6        # log(0)
            distrib_loss = -(Variable(target_prob) * soft_att_keys.log()).sum(-1).mean()

            # Add reconstruction loss
            re_loss = 0.5 * re_loss / args.batch_size
            bellman_loss = nn.SmoothL1Loss()(curr_qvalues, target_qvalues)
            # diversity_penalty = lambda_reg * diversity_penalty

            # loss = bellman_loss + distrib_loss + diversity_penalty + re_loss
            # loss = args.l1*distrib_loss + args.l2*bellman_loss + args.l3*diversity_penalty
            loss = args.l1 * distrib_loss + args.l2 * bellman_loss + args.l3 * diversity_penalty + args.l4 * re_loss

            # Backward
            loss.backward()

            # Clip gradients
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
                    torch.nn.utils.clip_grad_norm(layer.parameters(), args.grad_clip)

                if isinstance(layer, nn.Parameter):
                    torch.nn.utils.clip_grad_norm(layer, args.grad_clip)

            opt.step()

            loss_tracker.append(loss.data[0])
            # writer.add_scalar('bellman_loss', bellman_loss.data[0], timesteps)
            # writer.add_scalar('distributional_loss', distrib_loss.data[0], timesteps)
            # writer.add_scalar('diversity_loss', diversity_penalty.data[0], timesteps)
            # writer.add_scalar('reconstruction_loss', re_loss.data[0], timesteps)

        # if timesteps % args.test_freq == 0:
        #     mean_rewards_test = test(net, args)
        #     print('Avg. rewards(test): %.3f' % mean_rewards_test)
        #     writer.add_scalar('mean_rewards_test', mean_rewards_test, timesteps)

        if timesteps % args.print_freq == 0:
            print('Avg. loss: %.6f, Avg. cumulative reward: %.3f' % (np.mean(loss_tracker[-100:]),
                                                                     np.mean(cumulative_rewards_tracker[-100:])))
            print('Episodes completed: %d' % len(cumulative_rewards_tracker))

        if timesteps % args.sync_freq == 0:
            net_dict = net.state_dict()
            net_dict.pop('agg_values')
            tgt.load_state_dict(net_dict, strict=False)

        if timesteps % 2e6 == 0:
            # Halve the learning rate
            for param_group in opt.param_groups:
                param_group['lr'] /= 2.0
                param_group['lr'] = max(param_group['lr'], args.min_lr)

            print('LR: %.8f' % param_group['lr'])

    writer.close()


def test_one_episode(env, model, test_eps=0.001):
    state = env.reset()
    done = False
    tot_rewards = 0

    while not done:
        q_values, _, _ = model(Variable(ByteTensor(process_img(state)[None]), volatile=True))
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
