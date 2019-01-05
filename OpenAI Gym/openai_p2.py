import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter



LR = 1e-3


env = gym.make('MountainCar-v0')
env.reset()


goal_steps = 1000
score_requirements = -0.25
initial_games = 5000



def some_random_games_first():
    for episode in range(10):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = 2 # env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if done:
                break




def initial_population():
    training_data = []
    scores = []
    accepted_scores = []

    for _ in range(initial_games):
        score = -10
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0, 3)
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score = max(score, observation[0])

            if done:
                break


        if score >= score_requirements:
            accepted_scores.append(score)

            for data in game_memory:
                output = [0,0,0]
                output[data[1]] = 1

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)


    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print('Average accepted score', mean(accepted_scores))
    print('Median accepted score', median(accepted_scores))

    return training_data




def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 3, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')


    model = tflearn.DNN(network)

    return model




def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape([-1, len(training_data[0][0]), 1])
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=3, snapshot_step=500, show_metric=True,
              run_id='openai')

    return model



training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []

for each_game in range(10):
    score = 0
    game_memory = []
    prev_observation = []
    env.reset()
    for _ in range(1000):
        env.render()

        if len(prev_observation) == 0:
            action = random.randrange(0, 3)
        else:
            action = np.argmax(model.predict(prev_observation.reshape(-1, len(prev_observation), 1))[0])

        choices.append(action)

        observation, reward, done, info = env.step(action)
        prev_observation = observation
        game_memory.append([observation, action])
        score += reward

        if done:
            break

    scores.append(score)

print('Average score', sum(scores)/len(scores))
print('Choice 1: {} Choice 2: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))