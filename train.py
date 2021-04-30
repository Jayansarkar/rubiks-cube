import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Input, Dense, Flatten, Lambda, Add, Subtract
from tensorflow.keras.models import Sequential, Model

import numpy as np
import pickle
import os
from pathlib import Path
from tqdm import tqdm

from cube import CubeWrapper

TARGET_UPDATE_FREQ = 1000
TENSORBOARD_DIR = 'saved'
WRITE_TENSORBOARD = True


# config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8, allow_soft_placement=True,
#                         device_count={'CPU': 8})
#
# session = tf.compat.v1.Session(config=config)
# tf.config.set_soft_device_placement = True
# tf.config.threading.set_inter_op_parallelism_threads = 4
# tf.config.threading.set_intra_op_parallelism_threads = 8
# os.environ["OMP_NUM_THREADS"] = "6"
# os.environ["KMP_BLOCKTIME"] = "30"
# os.environ["KMP_SETTINGS"] = "1"
# os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"


def build_q_network(num_actions=12, input_shape=(54, 6), lr=0.0001):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(num_actions, activation='relu', kernel_initializer='he_uniform'))
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.mse)
    return model


# def build_q_network(num_actions, learning_rate=0.00001, input_shape=(84, 84)):
#     """Builds a dueling DQN as a Keras model
#     Arguments:
#         num_actions: Number of possible action the agent can take
#         learning_rate: Learning rate
#         input_shape: Shape of the preprocessed frame the model sees
#         history_length: Number of historical frames the agent can see
#     Returns:
#         A compiled Keras model
#     """
#     model_input = Input(shape=(input_shape[0], input_shape[1]))
#     x = Lambda(lambda layer: layer / 255)(model_input)
#
#     # x = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(
#     #     x)
#     # x = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(
#     #     x)
#     # x = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(
#     #     x)
#     # x = Conv2D(1024, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu',
#     #            use_bias=False)(x)
#     x = Dense(512, activation='relu', kernel_initializer=VarianceScaling(scale=2.), use_bias=False)
#
#     val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 3))(x)
#
#     val_stream = Flatten()(val_stream)
#     val = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_stream)
#
#     adv_stream = Flatten()(adv_stream)
#     adv = Dense(num_actions, kernel_initializer=VarianceScaling(scale=2.))(adv_stream)
#
#     reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))
#
#     q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])
#
#     model = Model(model_input, q_vals)
#     model.compile(tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.Huber())
#
#     return model


class ReplayBuffer:
    def __init__(self, size=10 ** 6, input_shape=(54, 6), use_per=True):
        self.size = size
        self.state_shape = input_shape
        self.count = 0
        self.index = 0

        self.states = np.empty((size, input_shape[0], input_shape[1]), dtype=np.bool)
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

    def save(self, file='replay.pkl'):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file):
        with open(file, 'rb') as f:
            return pickle.load(f)


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
                 final_epsilon=0.01,
                 max_timesteps=10 ** 6,
                 use_per=True):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.timestep = 0

        self.name = name
        self.dqn_path = os.path.join(self.save_folder, self.name, 'dqn.h5')
        self.target_dqn_path = os.path.join(self.save_folder, self.name, 'target_dqn.h5')
        self.replay_path = os.path.join(self.save_folder, self.name, 'replay.pkl')

        self.dqn = build_q_network(num_actions=num_actions)
        self.target_dqn = build_q_network(num_actions=num_actions)
        self.replay_buffer = ReplayBuffer(input_shape=self.input_shape)
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

        self.dqn.save(self.dqn_path)
        self.dqn.save(self.target_dqn_path)
        self.replay_buffer.save(self.replay_path)

    def load(self, load_replay_buffer=True):
        if not os.path.isdir(self.save_folder):
            raise ValueError(f'{self.save_folder} is not a valid directory')
        self.dqn = tf.keras.models.load_model(self.dqn_path)
        self.target_dqn = tf.keras.models.load_model(self.target_dqn_path)

        if load_replay_buffer:
            rb = ReplayBuffer.load(self.replay_path)
            self.replay_buffer = rb


dfa_rewards = {
    'daisy': 0.05,
    'white_cross_center': 0.1,
    'lower_level': 0.2,
    'middle_level': 0.3
}


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


def is_daisy(faces):
    return (faces['back'][[0, 2, 1, 1], [1, 1, 0, 1]] == 5).all()


def is_white_cross_center(faces):
    return is_daisy(faces) and faces['up'][0, 1] == faces['left'][0, 1] == faces['right'][0, 1] == faces['down'][
        0, 1] == 5


def is_lower_complete(faces):
    return ((faces['front'] == 5).all() and  # white face
            (faces['up'][2, :] == 3).all() and  # lower band
            (faces['left'][:, 2] == 0).all() and
            (faces['down'][0, :] == 4).all() and
            (faces['right'][:, 0] == 1).all())


def is_middle_complete(faces):
    return (faces['up'][1, :] == 3).all() and \
           (faces['left'][:, 1] == 0).all() and \
           (faces['down'][1, :] == 4).all() and \
           (faces['right'][:, 1] == 1).all()


def detect_conf(state):
    state_raw = tf.argmax(state, axis=1)
    faces = sep_faces(state_raw)
    if is_daisy(faces):
        return 1
    if is_white_cross_center(faces):
        return 2
    if is_lower_complete(faces):
        if is_middle_complete(faces):
            return 4
        return 3
    return 0


def automation_reward(new_obj_set, old_obj_set):
    new_obj = new_obj_set - old_obj_set
    if not new_obj:
        return 0
    new_obj = new_obj.pop()
    if 'middle_level' in old_obj_set:
        return 0
    if 'lower_level' in old_obj_set and new_obj != 'middle_level':
        return 0
    if 'white_cross_center' is old_obj_set and new_obj == 'daisy':
        return 0
    return dfa_rewards[new_obj]


if __name__ == '__main__':
    objs = ['daisy', 'white_cross_center', 'lower_level', 'middle_level']

    # TensorBoard writer
    writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

    main_agent = Agent()
    daisy_agent = Agent('daisy')
    white_cross_agent = Agent('white_cross_center')
    lower_level_agent = Agent('lower_level')
    middle_level_agent = Agent('middle_level')

    main_agent.dqn.summary()

    # main_agent.load()
    # daisy_agent.load()
    # white_cross_agent.load()
    # lower_level_agent.load()
    # middle_level_agent.load()

    env = CubeWrapper()

    timestep = eval_timestep = save_timestep = 0
    loss_list = []
    rewards = []

    pbar = tqdm(total=10 ** 7)
    try:
        with writer.as_default():
            while timestep < 10 ** 7:
                train_timestep = 0
                while train_timestep < 10 ** 6:  # and False:
                    active_agent = main_agent
                    detected_objs = set()
                    while detect_conf(env.reset()):
                        pass
                    start_st = np.argmax(env.getstate(), axis=1)
                    episode_reward_sum = 0
                    terminal = False
                    while not terminal:

                        timestep += 1
                        train_timestep += 1

                        eval_timestep = min(10 ** 5, eval_timestep + 1)
                        save_timestep = min(10 ** 6, save_timestep + 1)
                        pbar.update(1)

                        current_state = env.getstate()
                        action = active_agent.get_action(current_state)
                        new_state, reward, terminal, _ = env.step(action)

                        old_obj_set = detected_objs.copy()
                        conf = detect_conf(new_state)
                        if conf:
                            detected_objs.add(objs[conf - 1])
                        new_obj_set = detected_objs.copy()

                        if conf:
                            print(f'conf={conf}')
                        if conf == 1:
                            new_agent = daisy_agent
                        elif conf == 2:
                            new_agent = white_cross_agent
                        elif conf == 3:
                            new_agent = lower_level_agent
                        elif conf == 4:
                            new_agent = middle_level_agent
                        else:
                            new_agent = main_agent

                        ar = automation_reward(new_obj_set, old_obj_set)
                        if ar:
                            print(f'New: {new_obj_set}, Old:{old_obj_set}')
                            print(f'automation_reward;{ar}')
                            end_st = np.argmax(env.getstate(), axis=1)
                            env.set_state(start_st)
                            env.render()
                            env.set_state(end_st)
                            env.render()
                        reward = reward + 10 * ar
                        active_agent.add_experience(current_state, action, reward, terminal)

                        if active_agent.timestep > active_agent.replay_buffer_start_size:
                            loss, _ = active_agent.learn()
                            loss_list.append(loss)

                            if active_agent.timestep % TARGET_UPDATE_FREQ == 0:
                                active_agent.update_target_network()

                        episode_reward_sum += reward
                        active_agent = new_agent
                        detected_objs = new_obj_set
                    rewards.append(episode_reward_sum)

                    if len(rewards) % 10 == 0 and rewards:
                        # Write to TensorBoard
                        if WRITE_TENSORBOARD:
                            tf.summary.scalar('Reward', np.mean(rewards[-10:]), timestep)
                            if loss_list:
                                tf.summary.scalar('Loss', np.mean(loss_list[-100:]), timestep)
                            writer.flush()

                # save_timestep = 0
                main_agent.save()
                daisy_agent.save()
                white_cross_agent.save()
                lower_level_agent.save()
                middle_level_agent.save()

                # eval_timestep = 0
                # eval
                terminal = False
                eval_reward = 0
                env.reset()
                env.render()
                detected_objs = set()
                active_agent = main_agent
                while not terminal:
                    current_state = env.getstate()
                    action = active_agent.get_action(current_state, eval=True)
                    new_state, reward, terminal, _ = env.step(action)

                    old_obj_set = detected_objs.copy()
                    conf = detect_conf(new_state)
                    if conf:
                        detected_objs.add(objs[conf - 1])
                    new_obj_set = detected_objs.copy()

                    if conf == 1:
                        new_agent = daisy_agent
                    elif conf == 2:
                        new_agent = white_cross_agent
                    elif conf == 3:
                        new_agent = lower_level_agent
                    elif conf == 4:
                        new_agent = middle_level_agent
                    else:
                        new_agent = main_agent

                    reward = reward + 10 * automation_reward(new_obj_set, old_obj_set)
                    eval_reward += reward
                    active_agent = new_agent
                    detected_objs = new_obj_set
                    env.render()
                print(f'Evaluation reward: {eval_reward}')
    except KeyboardInterrupt:
        print('\nTraining exited early.')
        writer.close()
