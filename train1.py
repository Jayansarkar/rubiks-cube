import json
import os
import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tqdm import tqdm

from confs import str_to_state, state_to_str
from cube import CubeWrapper

TARGET_UPDATE_FREQ = 1000
TENSORBOARD_DIR = 'saved'
WRITE_TENSORBOARD = True


def build_q_network(num_actions=12, input_shape=(54,), lr=0.0001):
    model = Sequential()
    if not ONE_HOT_STATE:
        model.add(Dense(256, input_dim=54, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    else:
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(num_actions, activation='softmax', kernel_initializer='he_uniform'))
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.mse)
    return model


class ReplayBuffer:
    def __init__(self, size=10 ** 6, input_shape=(54, 6), use_per=True):
        self.size = size
        self.state_shape = input_shape
        self.count = 0
        self.index = 0

        self.states = np.empty((size,) + input_shape, dtype=np.int32)
        self.actions = np.empty(size, dtype=np.int32)
        self.rewards = np.empty(size, dtype=np.float32)
        self.terminals = np.empty(size, dtype=np.bool)
        self.priorities = np.zeros(self.size, dtype=np.float32)

        self.use_per = use_per

    def add_experience(self, state, action, reward, done, ):
        if self.state_shape != state.shape:
            raise ValueError('Dimension of state array is wrong!')

        self.actions[self.index] = action
        self.states[self.index] = state
        self.rewards[self.index] = reward
        self.terminals[self.index] = done

        self.priorities[self.index] = max(self.priorities.max(), 1)  # make the most recent experience important

        self.count = max(self.count, self.index + 1)
        self.index = (self.index + 1) % self.size

    def sample_minibatch(self, batch_size=32, priority_scale=0.0):
        if self.use_per:
            scaled_priorities = self.priorities[:self.count - 1] ** priority_scale
            sample_probabilities = scaled_priorities / sum(scaled_priorities)

        sampled_indices = np.empty(batch_size, dtype=np.int32)
        i = 0
        while i < batch_size:
            if self.use_per:
                ind = np.random.choice(np.arange(self.count - 1), p=sample_probabilities)
            else:
                ind = np.random.randint(0, self.count - 1)
            sampled_indices[i] = ind
            if not self.terminals[ind] and ind != (self.index - 1) % self.size:
                i += 1
        res = self.states[sampled_indices], self.actions[sampled_indices], self.rewards[sampled_indices], self.states[
            sampled_indices + 1], self.terminals[sampled_indices]
        if self.use_per:
            importance = 1 / self.count * 1 / sample_probabilities[sampled_indices]
            importance = importance / importance.max()
            return res, importance, sampled_indices
        return res

    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def save(self, save_dir='replay'):
        os.makedirs(save_dir, exist_ok=True)
        meta = {
            'count': self.count,
            'index': self.index
            # 'states': [state_to_str(st) for st in self.states],
            # 'actions': self.actions.tolist(),
            # 'rewards': self.rewards.tolist(),
            # 'terminals': self.terminals.tolist(),
            # 'priorities': self.priorities.tolist()
        }
        with open(f'{save_dir}/meta.json', 'w') as f:
            json.dump(meta, f)
        np.savez_compressed(f'{save_dir}/arrays',
                            states=self.states,
                            actions=self.actions,
                            rewards=self.rewards,
                            terminals=self.terminals,
                            priorities=self.priorities)

    def load(self, file):
        with open(f'{file}/meta.json') as f:
            meta = json.load(f)
        self.count = meta['count']
        self.index = meta['index']

        temp = np.load(f'{file}/arrays.npz')
        self.states = temp['states']
        self.actions = temp['actions']
        self.rewards = temp['rewards']
        self.terminals = temp['terminals']
        self.priorities = temp['priorities']


class Agent:
    save_folder = 'saved'

    def __init__(self, name='main',
                 input_shape=(54, 6),
                 batch_size=32,
                 gamma=0.95,
                 num_actions=12,
                 replay_buffer_start_size=10000,
                 initial_epsilon=1,
                 midway_epsilon=0.1,
                 midway_timestep=10 ** 5,
                 final_epsilon=0.001,
                 max_timesteps=10 ** 6,
                 use_per=True):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.timestep = 0

        self.name = name
        self.dqn_path = os.path.join(self.save_folder, self.name, 'dqn.h5')
        self.target_dqn_path = os.path.join(self.save_folder, self.name, 'target_dqn.h5')
        self.replay_path = os.path.join(self.save_folder, self.name, 'replay')
        self.timestep_path = os.path.join(self.save_folder, self.name, 'timestep')

        self.dqn = build_q_network(num_actions=num_actions, input_shape=self.input_shape)
        self.target_dqn = build_q_network(num_actions=num_actions, input_shape=self.input_shape)
        self.replay_buffer = ReplayBuffer(input_shape=self.input_shape, use_per=use_per)
        self.use_per = use_per

        self.batch_size = batch_size
        self.midway_timesteps = midway_timestep
        self.max_timesteps = max_timesteps
        self.replay_buffer_start_size = replay_buffer_start_size

        self.initial_epsilon = initial_epsilon
        self.midway_epsilon = midway_epsilon
        self.final_epsilon = final_epsilon

        self.slope1 = (self.midway_epsilon - self.initial_epsilon) / (
                self.midway_timesteps - self.replay_buffer_start_size)
        self.slope2 = (self.final_epsilon - self.midway_epsilon) / (self.max_timesteps - self.midway_timesteps)

    def get_epsilon(self, eval=False):
        if eval:
            return self.initial_epsilon
        if 0 < self.timestep - self.replay_buffer_start_size < self.midway_timesteps:
            return self.initial_epsilon + self.slope1 * (self.timestep - self.replay_buffer_start_size)
        if self.timestep >= self.replay_buffer_start_size + self.midway_timesteps:
            return self.midway_epsilon + self.slope2 * (self.timestep - self.midway_timesteps)
        return self.initial_epsilon

    def get_action(self, state, for_step=True, eval=False):
        if eval: for_step = False
        if for_step: self.timestep += 1
        epsilon = self.get_epsilon(eval=False)
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        q_vals = self.dqn.predict(np.array([state]))[0]
        return q_vals.argmax()

    def update_target_network(self):
        self.target_dqn.set_weights(self.dqn.get_weights())

    def add_experience(self, state, action, reward, done):
        self.replay_buffer.add_experience(state, action, reward, done)

    def learn(self, priority_scale=1.0):
        if self.use_per:
            (states, actions, rewards, new_states,
             terminals), importance, indices = self.replay_buffer.sample_minibatch(batch_size=self.batch_size,
                                                                                   priority_scale=priority_scale)
            importance = importance ** (1 - self.get_epsilon())
        else:
            states, actions, rewards, new_states, terminals = self.replay_buffer.sample_minibatch(
                batch_size=self.batch_size, priority_scale=priority_scale)
        # states, actions, rewards, new_states, terminals = self.replay_buffer.sample_minibatch(self.batch_size)

        argmax_next_q = self.dqn.predict(new_states).argmax(1)
        double_q = self.target_dqn.predict(new_states)[range(self.batch_size), argmax_next_q]
        target_q = rewards + self.gamma * double_q * (1 - terminals)

        with tf.GradientTape() as tape:
            q_values = self.dqn(states)
            one_hot_actions = tf.one_hot(actions, depth=self.num_actions)
            q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)
            error = q - target_q
            # loss = tf.keras.losses.Huber()(target_q, q)
            loss = tf.keras.losses.mse(target_q, q)
            if self.use_per:
                loss = tf.reduce_mean(loss * importance)

        model_gradients = tape.gradient(loss, self.dqn.trainable_variables)
        self.dqn.optimizer.apply_gradients(zip(model_gradients, self.dqn.trainable_variables))

        if self.use_per:
            self.replay_buffer.set_priorities(indices, error)

        return loss.numpy(), error

    def save(self):
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)
        for p in [self.dqn_path, self.target_dqn_path, self.replay_path]:
            p = Path(p)
            p.parent.mkdir(parents=True, exist_ok=True)
        with open(self.timestep_path, 'w') as f:
            f.write(f'{self.timestep}')

        self.dqn.save(self.dqn_path)
        self.target_dqn.save(self.target_dqn_path)
        self.replay_buffer.save(self.replay_path)

    def load(self, load_replay_buffer=True):
        if not os.path.isdir(self.save_folder):
            raise ValueError(f'{self.save_folder} is not a valid directory')
        self.dqn = tf.keras.models.load_model(self.dqn_path)
        self.target_dqn = tf.keras.models.load_model(self.target_dqn_path)
        with open(self.timestep_path) as f:
            self.timestep = int(f.read())

        if load_replay_buffer:
            self.replay_buffer.load(file=self.replay_path)


def sep_faces(state):
    faces = {
        'front': state[:9],
        'back': state[9:18],
        'up': state[36:45],
        'down': state[45:54],
        'left': state[27:36],
        'right': state[18:27]
    }
    return {k: np.array(v).reshape(3, 3) for k, v in faces.items()}


def load_confs():
    confs = {}
    for i in range(1, 7):
        with open(f'confs/away_{i}.pkl', 'rb') as f:
            confs[i] = pickle.load(f)
    return confs


def conf_and_automation_reward(state, confs, last_conf=None):
    state_str = state_to_str(state, one_hot=ONE_HOT_STATE)
    if last_conf is None:
        for i, conf in confs.items():
            if any([state_str in x for x in conf.values()]):
                return i
        return 0
    if last_conf == 0:
        if any([state_str in x for x in confs[6].values()]):
            return 6, 1
        return 0, 0
    if last_conf == 6:
        if any([state_str in x for x in confs[5].values()]):
            return 5, 1
        return 0, 0
    if last_conf - 1 in confs and any([state_str in x for x in confs[last_conf - 1].values()]):
        return last_conf - 1, 1
    if last_conf + 1 in confs and any([state_str in x for x in confs[last_conf + 1].values()]):
        return last_conf + 1, -1
    raise ValueError()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no-one-hot-state', '-noh', action='store_true')
    parser.add_argument('--use-priority', '-up', action='store_true')
    parser.add_argument('--iters', '-i', default=10 ** 7, type=int, help='Number of iterations')
    parser.add_argument('--min-shuffles', default=1, type=int, help='Minimum no of shuffles before each iterations')
    parser.add_argument('--max-shuffles', default=7, type=int, help='Maximum no of shuffles before each iterations')
    parser.add_argument('--max-steps', default=20, type=int, help='Maximum no of steps in each episode')
    args = parser.parse_args()

    ONE_HOT_STATE = not args.no_one_hot_state
    if ONE_HOT_STATE:
        INPUT_SHAPE = (54, 6)
    else:
        INPUT_SHAPE = (54,)
    USE_PRIORITY = args.use_priority
    MAX_ITERS = args.iters
    MAX_SHUFFLES = args.max_shuffles
    MAX_STEPS = args.max_steps

    objs = ['more_that_6'] + [f'away_{i}' for i in range(1, 7)]

    # TensorBoard writer
    writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

    main_agent = Agent(input_shape=INPUT_SHAPE, use_per=USE_PRIORITY)
    away_agents = [Agent(f'away_{i}', input_shape=INPUT_SHAPE, use_per=USE_PRIORITY) for i in range(1, 7)]
    agents = [main_agent] + away_agents

    main_agent.dqn.summary()

    confs = load_confs()

    try:
        for agent in agents:
            agent.load()
    except Exception as e:
        print(e)

    env = CubeWrapper(max_step=MAX_STEPS, one_hot_states=ONE_HOT_STATE)
    env.setScramble(1, MAX_SHUFFLES)

    timestep = 0
    loss_list = []
    rewards = []

    pbar = tqdm(total=MAX_ITERS)
    try:
        with writer.as_default():
            while timestep < MAX_ITERS:
                save_timestep = 0
                while save_timestep < 70000:
                    train_timestep = 0
                    while train_timestep < 30000:  # frames between eval and save

                        conf = conf_and_automation_reward(env.reset(), confs)
                        active_agent = agents[conf]
                        episode_reward_sum = 0
                        done = False

                        while not done:

                            timestep += 1
                            train_timestep += 1
                            save_timestep += 1

                            pbar.update(1)

                            current_state = env.getstate()
                            action = active_agent.get_action(current_state)
                            new_state, reward, done, _ = env.step(action)

                            if not done:
                                conf, ar = conf_and_automation_reward(new_state, confs, last_conf=conf)

                            else:
                                ar = 1
                            new_agent = agents[conf]

                            reward = reward + ar
                            active_agent.add_experience(current_state, action, reward, done)

                            if active_agent.timestep > active_agent.replay_buffer_start_size:
                                loss, _ = active_agent.learn()
                                loss_list.append(loss)

                                if active_agent.timestep % TARGET_UPDATE_FREQ == 0:
                                    active_agent.update_target_network()

                            episode_reward_sum += reward
                            active_agent = new_agent

                        rewards.append(episode_reward_sum)

                        if len(rewards) % 10 == 0 and rewards:
                            # Write to TensorBoard
                            if WRITE_TENSORBOARD:
                                tf.summary.scalar('Reward', np.mean(rewards[-10:]), timestep)
                                if loss_list:
                                    tf.summary.scalar('Loss', np.mean(loss_list[-100:]), timestep)
                                writer.flush()

                    # eval
                    done = False
                    eval_reward = 0
                    conf = conf_and_automation_reward(env.reset(), confs)
                    env.render()
                    detected_objs = [objs[conf]]
                    active_agent = agents[conf]
                    while not done:
                        current_state = env.getstate()
                        action = active_agent.get_action(current_state, eval=True)
                        new_state, reward, done, _ = env.step(action)

                        if not done:
                            conf, ar = conf_and_automation_reward(new_state, confs, last_conf=conf)
                            detected_objs.append(objs[conf])
                        else:
                            ar = 1

                        active_agent = agents[conf]

                        reward = reward + ar
                        eval_reward += reward
                        env.render()
                    print(f'Evaluation reward: {eval_reward}')
                    print(f'detected confs = {detected_objs}')

                # saving model
                for agent in agents:
                    agent.save()

    except KeyboardInterrupt:
        print('\nTraining exited early.')
        writer.close()
