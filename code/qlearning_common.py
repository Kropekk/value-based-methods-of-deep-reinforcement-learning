#!/usr/bin/env python3

__author__ = "Kamil Kropiewnicki"
import sys
import numpy as np
import tensorflow as tf
from collections import namedtuple
import logging
import cv2


class ConvolutionalLayers():
    #RGB_TO_GRAY_ARRAY = (0.299, 0.587, 0.114) 
    #STATE_SHAPE = (105, 80, 4) # Maybe worth investigating? see preprocess()
    STATE_SHAPE = (84, 84, 4)
    STATE_SIZE = int(np.prod(STATE_SHAPE))
    LAYER_DETAILS = ((32, 8, 4), (64, 4, 2), (64, 3, 1)) # (num_filters, kernel_size, stride)
    #param binary_frames (boolean): if true, set all non-black (0 value) pixels to 1. A bit of handcrafting features, may be useful for experiments.
    binary_frames = False # global variable because there is no point in passying it everywhere

    def __init__(self, scope_name):
        self.scope_name = scope_name
        with tf.variable_scope(scope_name) as scope:
            """None stands for BATCH_SIZE"""
            self.input = tf.placeholder(shape=(None, *self.STATE_SHAPE), dtype=tf.uint8, name="input") 
            self.ready_input = tf.cast(self.input, tf.float32) if self.binary_frames else self.input/255
            self.layer0 = tf.layers.conv2d(inputs=self.ready_input, filters=self.LAYER_DETAILS[0][0], kernel_size=self.LAYER_DETAILS[0][1],
                    strides=self.LAYER_DETAILS[0][2], padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())
            self.layer1 = tf.layers.conv2d(inputs=self.layer0, filters=self.LAYER_DETAILS[1][0], kernel_size=self.LAYER_DETAILS[1][1],
                    strides=self.LAYER_DETAILS[1][2], padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())
            self.layer2 = tf.layers.conv2d(inputs=self.layer1, filters=self.LAYER_DETAILS[2][0], kernel_size=self.LAYER_DETAILS[2][1],
                    strides=self.LAYER_DETAILS[2][2], padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())
            #self.final_layer = tf.layers.flatten(self.layer2)
            self.final_layer = tf.contrib.layers.flatten(self.layer2)  # use in case of old TF version

    @classmethod
    def preprocess(cls, data):
        """
        :param data: a single frame from the environment
        """ 
        """
        #    Manual attempt. Looks more naturally. However, it resizes to (105, 80) instead of (84, 84), which result in bigger state space. May be worth investingating.
        #    data = data[::2, ::2] # downsample
        #    data = np.dot(data, cls.RGB_TO_GRAY_ARRAY).astype(np.uint8) # convert to grayscale
        #    NOTE: There are known issues with tf.image.resize_image(), tf.image.resize_bilinear() etc. It is advised to stick to opencv library until tf.image is improved.
        """
        data = cv2.cvtColor(cv2.resize(data, (84, 84)), cv2.COLOR_RGB2GRAY) # typeof(data[0][0]): np.uint8 
        if cls.binary_frames:
            _, data = cv2.threshold(data,1,1,cv2.THRESH_BINARY) # use only 0 and 1 values (black pixels are left as 0, all the other ones as 1) 
        return data.reshape((*data.shape, 1))



class DQN():
    HIDDEN_LAYER_SIZE = 512
    def __init__(self, actions_len, input_layer, dueling):
        """
        :param actions_len (int):  env.action_space
        :param input_t: input for DQN, usually flattened output from convolutional layers
        :param dueling (boolean): if True use  dueling architecture; pure DQN otherwise
        """
        ### Does it ever happen that one input_layer (conv_net) is used as part of many DQNs? If so, different scope_name should be possible
        with tf.variable_scope(input_layer.scope_name) as scope:
            self.actions_len = actions_len
            self.input_layer = input_layer
            self.hidden_layer0 = tf.layers.dense(inputs=self.input_layer.final_layer, units=self.HIDDEN_LAYER_SIZE, activation=tf.nn.relu, 
                    kernel_initializer=tf.glorot_uniform_initializer(), bias_initializer=tf.glorot_uniform_initializer())
            if not dueling: # Q(s,a)
                self.final_layer = tf.layers.dense(inputs=self.hidden_layer0, units=self.actions_len, activation=None, 
                        kernel_initializer=tf.glorot_uniform_initializer(), bias_initializer=tf.glorot_uniform_initializer())
            else: # Q(s, a) = V(s) + (A(s,a) - average(A(s,a))) = V(s) - average(A(s,a)) + A(s,a)
                self.V = tf.layers.dense(inputs=self.hidden_layer0, units=1, activation=None, 
                        kernel_initializer=tf.glorot_uniform_initializer(), bias_initializer=tf.glorot_uniform_initializer())
                self.A = tf.layers.dense(inputs=self.hidden_layer0, units=self.actions_len, activation=None, 
                        kernel_initializer=tf.glorot_uniform_initializer(), bias_initializer=tf.glorot_uniform_initializer())
                self.final_layer = self.V - tf.reduce_mean(self.A, axis=1, keep_dims=True) + self.A # NOTE: keep_dims is deprecated, use keepdims instead


class NeuralNetwork():
    #RMSPropParameters = (0.00025, 0.99, 0.95, 1e-2) # Mnih et al. (2015) ? 
    RMSPropParameters = (0.00025, 0.99, 0.0, 1e-6) 
    def __init__(self, value_net, use_double_dqn):
        """
        :param value_net: DQN object
        :param use_double_dqn (bool): If true, use Double DQN algorithm.
        """
        with tf.variable_scope(value_net.input_layer.scope_name) as scope:
            self.use_double_dqn = use_double_dqn
            self.value_net = value_net
            self.rewards = tf.placeholder(shape=(None), dtype=tf.float32, name="rewards")
            self.not_terminals = tf.placeholder(shape=(None), dtype=tf.bool, name="not_terminals")
            self.predictions_for_next_state = tf.placeholder(shape=(None, self.value_net.actions_len), dtype=tf.float32, name="predictions_for_next_state") # from target network
            self.discount = tf.placeholder(shape=(), dtype=tf.float32, name="discount")

            """target = reward + discount * max(Q(s', a)) if not terminal else reward. See PDF for more info"""
            if self.use_double_dqn:
                self.predictions_for_next_state_from_online_network = tf.placeholder(shape=(None, self.value_net.actions_len),
                        dtype=tf.float32, name="predictions_for_next_state_from_online_netowrk")
                
                # In new TF casting shouldn't be necessary, see https://github.com/tensorflow/tensorflow/issues/8951
                self.best_actions_from_online_network = tf.cast(tf.argmax(self.predictions_for_next_state_from_online_network, axis=1), dtype=tf.int32)  

                """To select desired outputs, create array of size [batch_size], where each element is equal to: batch_number * available_actions.n + selected_action_in_given_batch"""
                self.actions_from_online_network = tf.range(tf.shape(self.best_actions_from_online_network)[0]) * self.value_net.actions_len  + self.best_actions_from_online_network
                self.targets = self.rewards + (tf.cast(self.not_terminals, tf.float32) * self.discount * tf.gather(
                    tf.reshape(self.predictions_for_next_state, [-1]), self.actions_from_online_network))
            else:
                self.targets = self.rewards + (tf.cast(self.not_terminals, tf.float32) * self.discount * tf.reduce_max(self.predictions_for_next_state, axis=1))
            
            self.chosen_actions = tf.placeholder(shape=(None), dtype=tf.int32, name="actions")

            """To select desired outputs, create array of size [batch_size], where each element is equal to: batch_number * available_actions.n + selected_action_in_given_batch"""
            self.indices = tf.range(tf.shape(self.chosen_actions)[0]) * self.value_net.actions_len  + self.chosen_actions

            """Convert tensor [batch_size, available_actions.n] into [batch_size * available_actions.n], then pick corresponding actions""" 
            self.actions_values = tf.gather(tf.reshape(self.value_net.final_layer, [-1]), self.indices)

            self.cost = tf.losses.huber_loss(labels=self.targets, predictions=self.actions_values, reduction=tf.losses.Reduction.MEAN)
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.RMSPropParameters[0], decay=self.RMSPropParameters[1],
                    momentum=self.RMSPropParameters[2], epsilon=self.RMSPropParameters[3])
            self.train = self.optimizer.minimize(self.cost, global_step=tf.train.get_or_create_global_step())

            """Allow copying"""
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
            self.trainable_vars_by_name = {var.name[len(scope.name):]: var for var in self.vars}
            self.trainable_vars_by_name_with_scope= {var.name: var for var in self.vars}


    def copy_weights_from_source(self, session, source):
        """
        :param session: tf.Session() object
        :param source: NeuralNetwork object with the same architecture as self
        """
        logging.info("copy_weights_from_source")
        tf.group(*[tensor.assign(source.trainable_vars_by_name[name]) for name, tensor in self.trainable_vars_by_name.items()]).run(session=session)
            

    def estimate_q_s(self, session, states):
        """
        Estimate Q(s,a) for each action in env.action_space
        :param session: tf.Session() object
        :param states: batch of ReplayMemory.Experience. states.shape == [batch_size, *ConvolutionalLayers.STATE_SHAPE]
        :return: [batch_size, env.action_space.n]
        """
        return session.run(self.value_net.final_layer, feed_dict={self.value_net.input_layer.input:states}) 

    def update(self, session, minibatch, discount, target_network):
        """
        Update weights of NN.
        :parram session: tf.Session()
        :param minibatch: minibatch of ReplayMemory.Experience tuples 
        :param discount: discount factor, also known as gamma
        :param target_network: Target network. If not used, then pass target_network = self
        """
        states, actions, rewards, nextstates, not_terminals = zip(*minibatch)
        next_state_estimations = target_network.estimate_q_s(session, nextstates)
        feed_dict={self.value_net.input_layer.input:states,  self.chosen_actions: actions,
            self.rewards: rewards, self.not_terminals: not_terminals, self.discount: discount, self.predictions_for_next_state: next_state_estimations}
        if self.use_double_dqn:
            next_state_estimations_from_online_network = self.estimate_q_s(session, nextstates)
            feed_dict[self.predictions_for_next_state_from_online_network] = next_state_estimations_from_online_network
        session.run(self.train, feed_dict=feed_dict)
        

class ReplayMemory():
    Experience = namedtuple("Experience", ["s", "a", "r", "s_", "not_terminal"]) # NOTE: we store inversion of done for convenience purposes! 
    def __init__(self, max_size, clip_reward=True):
        self.storage = []
        self.counter = 0
        self.max_size = max_size
        self.clip_reward = clip_reward

    def get_random_minibatch(self, minibatch_size):
#        return np.random.choice(self.storage, size=minibatch_size) # np.random.choice works only for 1-d arrays. However, this behaviour should be updated in the future.
        indices = np.random.randint(len(self.storage), size=minibatch_size)
        return [self.storage[i] for i in indices]


    def add_experience(self, state, action, reward, next_state, done):
        exp = self.Experience(state, action, np.sign(reward) if self.clip_reward else reward, next_state, not done)
        if self.counter < self.max_size:
            self.storage.append(exp)
        else:
            self.storage[self.counter % self.max_size] = exp 
        self.counter = self.counter + 1

    def populate(self, steps, policy_fn, env, starting_state=None):
        logging.info("Populate replay memory with {} experiences".format(steps))
        if starting_state is None:
            state = create_state_from_one_frame(env.reset())
        else:
            state = starting_state
        for _ in range(steps):
            state, _, done = make_step(policy_fn, env, state, self)
            if done:
                state = create_state_from_one_frame(env.reset())
        logging.info("Populating finished")



def create_next_state(prev_state, new_frame):
    split = np.dsplit(prev_state, 4)
    return np.dstack([split[1], split[2], split[3], ConvolutionalLayers.preprocess(new_frame)])

def create_state_from_one_frame(frame):
    return np.dstack([ConvolutionalLayers.preprocess(frame)] * 4)

def eps_generator(initial_exploration, final_exploration, final_exploration_frame):
    factor = (initial_exploration-final_exploration)/final_exploration_frame
    for i in range(final_exploration_frame):
        yield initial_exploration - i*factor
    while True:
        yield final_exploration

def create_policy(name, action_space_n, **kwargs):
    logging.info("create_policy {}".format(name))
    if name == "eps-greedy":
        eps = eps_generator(kwargs['initial_eps'], kwargs['final_eps'], kwargs['final_exploration_frame'])
        def eps_greedy_policy(state):
            if np.random.uniform() < next(eps): # choose random action
                return np.random.randint(action_space_n)
            action_values = kwargs['dqn'].estimate_q_s(kwargs['session'], [state])
            return np.argmax(action_values)
        return eps_greedy_policy
    elif name == "random":
        def random_policy(state):
            return np.random.randint(action_space_n)
        return random_policy
    else:
        raise ValueError("Choose appropriate name!")


def make_step(policy_fn, env, state, replay_memory=None):
    action = policy_fn(state)
    if "NoFrameskip" in env.unwrapped._spec.id: # perform 4 steps
        frame, reward, done, _ = env.step(action)
        prev_frame = frame # in case done == True
        for _ in range(3):
            if done:
                break
            prev_frame = frame
            frame, r1, d1, _ = env.step(action)
            reward = reward + r1
            done = done or d1
        next_state = create_next_state(state, np.maximum(prev_frame, frame))
    else:
        next_frame, reward, done, _ = env.step(action)
        next_state = create_next_state(state, next_frame)
    if replay_memory is not None:
        replay_memory.add_experience(state, action, reward, next_state, done)
    return next_state, reward, done 

def evaluate(policy_fn, env, eval_num, updates_num, replay_memory=None):
    logging.info("Start evaluation number {}".format(eval_num))
    state = create_state_from_one_frame(env.reset())
    frame_counter = 0
    total_reward = 0
    done = False
    while not done:
        state, reward, done = make_step(policy_fn, env, state, replay_memory)
        frame_counter = frame_counter + 1
        total_reward = total_reward + reward
    logging.info("In evaluation number {} agent obtained {} total reward after {} frames. SGD was perfomed {} times so far".format(eval_num, total_reward, frame_counter, updates_num))
    return total_reward

