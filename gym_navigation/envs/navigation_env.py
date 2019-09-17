

import numpy as np
import time
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_maze.envs.maze_view_2d import MazeView2D


class Nav_Env(gym.Env):

    SHIFT = {0 : [1,0],
             1 : [-1,0],
             2 : [0,1],
             3 : [0,-1]}

    def __init__(self, env_size = 10, obstacles = 0):
        self.MAX_STEPS = 20

        self.size = env_size
        self.obstacles = obstacles

        self.actions = ['up', 'down', 'left', 'right']

        self.matrix = np.zeros([self.size] * 2)

        for _ in range(self.obstacles):
            x, y = np.random.randint(self.size, size=2)
            self.matrix[x,y] = 1

        # WALLS
        self.matrix[:, 0] = self.matrix[0, :] = self.matrix[:, -1] = self.matrix[-1, :] = 1

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_state(self):

        player_pos = np.zeros([self.size] * 2)
        goal_pos = np.zeros([self.size] * 2)

        player_pos[tuple(self.player_position)] = 1
        goal_pos[tuple(self.goal_position)] = 1

        state = np.array([self.matrix, player_pos, goal_pos])

        return state

    def reset(self):

        self.step_count = 0
        self.player_position = [0, 0]
        self.goal_position = [0, 0]

        while self.matrix[tuple(self.player_position)] == 1:
            self.player_position = np.random.randint(self.size, size=2)

        while self.matrix[tuple(self.goal_position)] == 1:
            self.goal_position = np.random.randint(self.size, size=2)

        self.obs_shape = [3, self.size, self.size]
        self.action_space = spaces.MultiDiscrete(len(self.actions))

        # The observation space is actually composed of a vector of twice the length of the guesses
        # The first half contains the guess that we have just made
        # The second half is the result of the query (containing 2,1,0) that indicates how many are correct
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.int)

        return self._get_state()

    def step(self, action):
        assert self.action_space.contains(action)

        new_player_position = self.player_position + self.SHIFT[action]

        if self.matrix[tuple(new_player_position)] == 0:
            self.player_position = new_player_position

        done = (self.goal_position == self.player_position).all()
        reward = 1 if done else -0.05

        if self.step_count > self.MAX_STEPS:
            done = True

        self.step_count += 1

        return self._get_state(), reward, done, None

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self, mode='human', close=False):
        view = np.copy(self.matrix).astype(np.str)
        view[tuple(self.player_position)] = 'p'
        view[tuple(self.goal_position)] = 'g'

        view[view=='0.0'] = '.'
        view[view=='1.0'] = '+'

        for row in view:
            for char in row:
                print(char, end="")
            print()

    def close(self):
        return


if __name__ == "__main__":
    import time

    env = Nav_Env(10)
    _ = env.reset()

    print(env.action_space)

    for _ in range(0,100):
        _, reward, done, _ = env.step(env.action_space.sample())
        env.render()

        if done:
            env.reset()

        time.sleep(0.1)
