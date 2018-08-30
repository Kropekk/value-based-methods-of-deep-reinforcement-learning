#!/usr/bin/env python3

__author__ =  "Kamil Kropiewnicki"
import sys
import gym
import qlearning_common as qc
import tensorflow as tf
import os
import logging
import argparse

#####################
#                   #
#  HYPERPARAMATERS  #
# Mnih et al.(2015) #
#                   #
#####################
MINIBATCH_SIZE = 32
REPLAY_MEMORY_SIZE = int(3e5) #int(1e6) in the Mnih et al. (2015) # 3e5 breakout, 1e5 spaceinvaders
DISCOUNT = 0.99
SGD_UPDATE_FREQ = 4 # NOTE: Despite SGD in variable name, we use different optimization
INITIAL_EXPLORATION = 1.0
FINAL_EXPLORATION = 0.1
EVAL_EXPLORATION = 0.05
TARGET_NETWORK_UPDATE_FREQ = int(1e4)
PRELEARNING_STEPS = int(5e4) # replay_memory initialization size
FINAL_EXPLORATION_FRAME = int(1e6)
FRAMES_LIMIT = int(1e7) # 5e7 (2e8 in Double DQN paper)
#NO_OP_MAX = 30
#####################
EVAL_FREQ = 50

class DqnAgent():
    training_names = ["simple_dqn", "target_dqn", "ddqn", "dueling_dqn", "dueling_ddqn"]

    def __init__(self, args):
        self.env = gym.make(args.env)
        self.clip_reward = not args.dont_clip_reward
        self.train_str_info = "{}_binary_frames_{}_clip_reward_{}".format(args.training_name, args.binary_frames, self.clip_reward)
        
        self.saver_path = os.path.join(os.path.abspath("./{}/models/{}/".format(args.env, self.train_str_info)), "")
        self.tensorboard_save_path = os.path.join(os.path.abspath("./{}/tensorboard/{}/".format(args.env, self.train_str_info)), "")
        self.eval_path = os.path.join(os.path.abspath("./{}/videos/{}/".format(args.env, self.train_str_info)), "")
        os.makedirs(self.saver_path, exist_ok=True) # mkdir -p

        self.use_periodically_updated_target_network = args.training_name != "simple_dqn"
        self.use_double_dqn = "ddqn" in args.training_name
        self.use_dueling_architecture = "dueling" in args.training_name

        self.config_for_tf_session = tf.ConfigProto(device_count = {'GPU': 1})
        qc.ConvolutionalLayers.binary_frames = args.binary_frames

        self.dqn = qc.NeuralNetwork(qc.DQN(actions_len=self.env.action_space.n, input_layer=qc.ConvolutionalLayers("ONLINE_NETOWRK"),
            dueling=self.use_dueling_architecture), use_double_dqn=self.use_double_dqn)
        if self.use_periodically_updated_target_network and not args.evaluate:  # no need for target network during evaluation
            self.dqn_target = qc.NeuralNetwork(qc.DQN(actions_len=self.env.action_space.n, input_layer=qc.ConvolutionalLayers("TARGET_NETWORK"),
                dueling=self.use_dueling_architecture), use_double_dqn=self.use_double_dqn)


        self.reward_placeholder = tf.placeholder('float')
        tf.summary.scalar('Reward during evaluation', self.reward_placeholder)
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()
    
    def train(self):
        os.makedirs(self.tensorboard_save_path, exist_ok=True) # mkdir -p
        with tf.Session(config=self.config_for_tf_session) as sess:
            if tf.train.latest_checkpoint(self.saver_path):
                logging.info("Restore previously trained model from {}".format(self.saver_path))
                self.saver.restore(sess, self.saver_path)
                logging.info("SGD have been performed {} times so far".format(sess.run(tf.train.get_global_step())))
            else:
                logging.info("Train from the scratch")
                sess.run(tf.global_variables_initializer())
            eps_greedy_policy = qc.create_policy("eps-greedy", self.env.action_space.n, initial_eps=INITIAL_EXPLORATION,
                    final_eps=FINAL_EXPLORATION, final_exploration_frame=FINAL_EXPLORATION_FRAME, dqn=self.dqn, session=sess)
            random_policy = qc.create_policy("random", self.env.action_space.n)
            eps_greedy_evaluation_policy = qc.create_policy("eps-greedy", self.env.action_space.n, initial_eps=EVAL_EXPLORATION,
                    final_eps=EVAL_EXPLORATION, final_exploration_frame=1, dqn=self.dqn, session=sess)
            replay_memory = qc.ReplayMemory(REPLAY_MEMORY_SIZE, clip_reward=self.clip_reward)
            replay_memory.populate(PRELEARNING_STEPS, random_policy, self.env)
            if self.use_periodically_updated_target_network: 
                self.dqn_target.copy_weights_from_source(sess, self.dqn)
            episode = 0
            max_reward = -1.0
            writer = tf.summary.FileWriter(self.tensorboard_save_path, sess.graph)
            while replay_memory.counter < FRAMES_LIMIT:
                logging.info("{}: {}% of training done".format(self.train_str_info, (replay_memory.counter/FRAMES_LIMIT)*100.0))
                episode = episode + 1
                if (episode % EVAL_FREQ) == 0:
                    reward_in_eval = qc.evaluate(eps_greedy_evaluation_policy, self.env, episode//EVAL_FREQ, sess.run(tf.train.get_global_step()))
                    max_reward = max(max_reward, reward_in_eval)
                    writer.add_summary(sess.run(self.summary_op, feed_dict={self.reward_placeholder:reward_in_eval}), episode//EVAL_FREQ)
                    self.saver.save(sess, self.saver_path)
                    logging.info("Model saved. Reward obtained in eval: {}. Max reward: {}. Replay memory holds {} experiences".format(reward_in_eval, max_reward, len(replay_memory.storage)))

                state = qc.create_state_from_one_frame(self.env.reset())
                done = False
                while not done: # it's never endless loop due to env._max_episode_steps

                    """Populate replay memory"""
                    for _ in range(SGD_UPDATE_FREQ):
                        state, _, done = qc.make_step(eps_greedy_policy, self.env, state, replay_memory)
                        if done:
                            break

                    """Experience replay"""
                    self.dqn.update(sess, replay_memory.get_random_minibatch(MINIBATCH_SIZE), DISCOUNT, target_network=self.dqn_target if self.use_periodically_updated_target_network else self.dqn)

                    """Update weights of target network"""
                    if self.use_periodically_updated_target_network and sess.run(tf.train.get_global_step()) % TARGET_NETWORK_UPDATE_FREQ == 0: # global_step is incremented only after perfoming SGD.
                        self.dqn_target.copy_weights_from_source(sess, self.dqn)

            if self.use_periodically_updated_target_network: 
                self.dqn_target.copy_weights_from_source(sess, self.dqn)
            self.saver.save(sess, self.saver_path)
            reward_in_eval = qc.evaluate(eps_greedy_evaluation_policy, self.env, (episode//EVAL_FREQ)+1, sess.run(tf.train.get_global_step()))
            max_reward = max(max_reward, reward_in_eval)
            writer.close()
            logging.info("Model saved. Reward obtained in the last eval: {}. Max reward: {}. Replay memory holds {} experiences".format(reward_in_eval, max_reward, replay_memory.counter))
            reward_file = open(os.path.join(self.saver_path, "max_reward.txt"), 'a')
            reward_file.write("Max reward obtained during training: {}".format(max_reward))
            reward_file.close()
            logging.info("Finished, max reward in eval: {}".format(max_reward))

    def evaluate(self, episodes):
        os.makedirs(self.eval_path, exist_ok=True) # mkdir -p
        env = gym.wrappers.Monitor(self.env, directory=self.eval_path, video_callable = lambda episode_id: True, resume=True)
        max_reward = -1
        id_max_reward = -1
        with tf.Session(config=self.config_for_tf_session) as sess:
            if tf.train.latest_checkpoint(self.saver_path):
                logging.info("Restore previously trained model from {}".format(self.saver_path))
                self.saver.restore(sess, self.saver_path)
            else:
                raise Exception("NO MODEL. ABORT")
            eps_greedy_evaluation_policy = qc.create_policy("eps-greedy", self.env.action_space.n, initial_eps=EVAL_EXPLORATION,
                    final_eps=EVAL_EXPLORATION, final_exploration_frame=1, dqn=self.dqn, session=sess)
            writer = tf.summary.FileWriter(self.eval_path, sess.graph)
            for i in range(episodes):
                reward_in_eval = qc.evaluate(eps_greedy_evaluation_policy, env, i, sess.run(tf.train.get_global_step()))
                writer.add_summary(sess.run(self.summary_op, feed_dict={self.reward_placeholder:reward_in_eval}), i)
                if reward_in_eval > max_reward:
                    max_reward = reward_in_eval
                    id_max_reward = i
                    logging.info("New best evaluation!")
            logging.info("Finished! Best achieved reward: {} during {} evaluation.".format(max_reward, id_max_reward))
            writer.close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN or its variant on one of OpenAI gym game.")
    parser.add_argument("--env", help="Environment name available in OpenAI gym, for example BreakoutDeterministic-v4", required=True)
    parser.add_argument("--training_name", help="Training name", required=True, choices=DqnAgent.training_names)
    parser.add_argument("--GPU", help="(comma-seperated) IDs of GPUs to use. Recommended to use because of significant speed-up."
            " The value is passed to CUDA_VISIBLE_DEVICES", default="")
    parser.add_argument("--evaluate", help="If used, perform evaluation instead of training. Pass number of evaluations to run. Videos will be saved.", type=int)
    parser.add_argument("--binary_frames", help="If used, all non-black pixels in frame are set to value 1. Might make training easier.", action="store_true")
    parser.add_argument("--dont_clip_reward", help="If used, reward obtained from environment is not clipped to [-1; 1]." 
            "Note that Huber loss (gradient clipping) is always present.", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU

    dqn_agent = DqnAgent(args)
    if args.evaluate:
        dqn_agent.evaluate(args.evaluate)
    else:
        dqn_agent.train()

