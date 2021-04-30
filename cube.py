import tensorflow as tf
from gym_Rubiks_Cube.envs.rubiks_cube_env import RubiksCubeEnv, tileDict, actionList
from gym_Rubiks_Cube.envs import cube


class CubeWrapper(RubiksCubeEnv):
    def __init__(self, orderNum=3, one_hot_states=True, max_step=40):

        super().__init__(orderNum)
        self.ncube = cube.Cube(order=self.orderNum)
        self.step_count = 0
        self.action_log = []
        self.one_hot_states = one_hot_states
        self.state = self.getstate()
        self.max_step = max_step

    def getstate(self):
        state = super(CubeWrapper, self).getstate()
        if self.one_hot_states:
            return tf.one_hot(state, depth=len(tileDict), dtype=tf.int8)
        return state

    def step(self, action):
        self.action_log.append(action)
        self.ncube.minimalInterpreter(actionList[action])
        self.state = self.getstate()
        self.step_count = self.step_count + 1

        reward = 0.0
        done = False
        others = {}
        if self.ncube.isSolved():
            reward = 10.0
            done = True

        if self.step_count > self.max_step:
            done = True

        return self.state, reward, done, others

    def set_state(self, state):
        self.ncube.destructVectorState(state, inBits=True)
