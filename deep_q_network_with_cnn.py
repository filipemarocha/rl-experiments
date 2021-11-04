import gym
import gym_simple
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential, clone_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from skimage.transform import resize

env = gym.make('PutBallInBoxEnvRandom5x5-v0')

train_episodes = 1000         # max number of episodes to learn from
max_steps = 1000              # max steps in an episode
gamma = 0.95                  # future reward discount

# Exploration parameters
explore_start = 1.0           # exploration probability at start
explore_stop = 0.1            # minimum exploration probability
decay_rate = 0.00005          # exponential decay rate for exploration prob

# Network parameters
hidden_size = 100             # number of units in each Q-network hidden layer
learning_rate = 0.0001        # Q-network learning rate

# Memory parameters
memory_size = 50000           # memory capacity
batch_size = 5000           # experience mini-batch size
pretrain_length = batch_size  # number experiences to pretrain the memory

# Update target Network
c_steps = 100                  # every c steps

# Each grid cell has 120x120 pixels
state_size = 80,80,3

# Actions: Move up, down, left, right, up-right, up-left, down-right, down-left, pick-up, put-down
action_size = 10

class QNetwork:
    def __init__(self, state_size, action_size, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.state_size))
        #model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        #model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model

from collections import deque

class Memory:
    def __init__(self, max_size = 1000):
        self.buffer = deque(maxlen=max_size)
    # experience: (state, action, reward, next_state)
    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]


# Initialize the simulation
_, reward, done, _ = env.reset()
state = env.render()
state = resize(state,(80,80))

memory = Memory(max_size=memory_size)

# Make a bunch of random actions and store the experiences
for ii in range(pretrain_length):
    # Uncomment the line below to watch the simulation
    #env.render()
    # Make a random action
    action = env.action_space.sample()
    _, reward, done, _ = env.step(action)
    next_state = env.render()
    next_state = resize(next_state,(80,80))

    if done:
        # The simulation fails so no next state
        #next_state = np.zeros(state.shape)
        # Add experience to memory
        memory.add((state, action, reward, next_state))

        # Start new episode
        _, reward, done, _ = env.reset()
        state = env.render()
        state = resize(state,(80,80))
    else:
        # Add experience to memory
        memory.add((state, action, reward, next_state))
        state = next_state


mainQN = QNetwork(state_size, action_size, learning_rate).build_model()
mainQN.summary()
targetQN = clone_model(mainQN)
targetQN.summary()
# update target
targetQN.set_weights(mainQN.get_weights())


def main():
    cum_avg_reward_lst = []
    step = 0
    total_reward = 0
    for ep in range(1, train_episodes):
        _, reward, done, _ = env.reset()
        state = env.render()
        state = resize(state,(80,80))
        print('Episode: ' + str(ep))
        t = 0
        while t < max_steps:
            step += 1
            # Uncomment this next line to watch the training
            #env.render()

            # Explore or Exploit
            explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step)
            if explore_p > np.random.rand():
                # Make a random action
                action = env.action_space.sample()
            else:
                # Get action from Q-network
                act_values = mainQN.predict(state.reshape(1,80,80,3))
                action = np.argmax(act_values[0])

            # Take action, get new state and reward
            _, reward, done, _ = env.step(action)
            next_state = env.render()
            next_state = resize(next_state,(80,80))

            total_reward += reward

            if done:
                # the episode ends so no next state
                #next_state = np.zeros(state.shape)
                print("Reached the goal after {} timesteps".format(t))
                """
                print('Episode: {}'.format(ep),
                      'Training loss: {:.4f}'.format(loss),
                      'Explore P: {:.4f}'.format(explore_p))
                """
                # Add experience to memory
                memory.add((state, action, reward, next_state))
            else:
                # Add experience to memory
                # If is the last step in the episode and goal isnt Reached
                # return reward 100, since final state is emulating goal
                # if t+1 == max_steps:
                memory.add((state, action, reward, next_state))
                state = next_state
                t += 1

            # Sample mini-batch from memory
            batch = memory.sample(batch_size)
            states = np.array([each[0] for each in batch])
            actions = np.array([each[1] for each in batch])
            rewards = np.array([each[2] for each in batch])
            next_states = np.array([each[3] for each in batch])

            # Train network
            target_Qs = targetQN.predict(next_states)
            # Set target_Qs to 0 for states where episode ends
            #episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
            #target_Qs[episode_ends] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

            targets = rewards + gamma * np.amax(target_Qs,axis=1)
            #targets.shape
            # * np.max(target_Qs, axis=1)
            # make the agent to approximately map
            # the current state to future discounted reward
            # We'll call that target_f
            # Train the Neural Net with the state and target_f
            targets_f = targetQN.predict(states)
            targets_f[:,actions]=targets
            mainQN.fit(states, targets_f, epochs=1, verbose=0)
            # After every C steps copy Q network to target network periodically
            # use target network to calculate target Qs
            if t % c_steps == 0:
                targetQN.set_weights(mainQN.get_weights())
            if done:
                break
        cum_avg_reward = total_reward/ep
        cum_avg_reward_lst.append(cum_avg_reward)
        print(cum_avg_reward)
    mainQN.save("putballinbox_dqn_model")

main()
