'''
Q-learning approach for different RL problems
as part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning
 
Inspired by https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA
 
        @author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''
import random
import os
import pickle

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, value):

        oldv = self.getQ(state, action)
        self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state):
        q = [self.getQ(state, action) for action in self.actions]

        if random.random() < self.epsilon:
            action = random.choice(list(self.actions))
        else:
            max_q = max(q)
            best_actions = [
                action
                for action, q_value in zip(self.actions, q)
                if q_value == max_q
            ]
            action = random.choice(best_actions)

        return action

    def learn(self, state1, action1, reward, state2, done=False):
        if done:
            target = reward
        else:
            maxqnew = max([self.getQ(state2, action) for action in self.actions])
            target = reward + self.gamma * maxqnew
        self.learnQ(state1, action1, target)

    def save_model(self, file_path):
        """Save the learned Q-table and training parameters."""

        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        model = {
            'q_table': self.q,
            'actions': list(self.actions),
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'gamma': self.gamma,
        }

        temporary_path = file_path + '.tmp'

        with open(temporary_path, 'wb') as model_file:
            pickle.dump(
                model,
                model_file,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        os.replace(temporary_path, file_path)