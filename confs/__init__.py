from cube import CubeWrapper, actionList
import numpy as np

from collections import defaultdict
import pickle
from pathlib import Path


def is_inverse(action1, action2):
    # action1, action2 = int(action1), int(action2)
    if actionList[action1] == '.' + actionList[action2] or '.' + actionList[action1] == actionList[action2]:
        return True
    return False


def str_to_state(state_str):
    return [int(x) for x in state_str]


def state_to_str(state, one_hot=False):
    if one_hot:
        state = np.argmax(state, axis=1)
    return ''.join(map(str, state))


def gen_confs(away=1, path='away', save=True):
    res = defaultdict(set)
    cube_env = CubeWrapper(one_hot_states=False)
    cube_env.setScramble(1, 2, doScamble=False)
    if away == 1:
        for action in range(cube_env.action_space.n):
            cube_env.reset()
            cube_env.step(action)

            res[action].add(state_to_str(cube_env.getstate()))
    else:
        p = Path(f'{path}_{away - 1}.pkl')
        if p.exists():
            with open(p, 'rb') as f:
                prev_states = pickle.load(f)
        else:
            prev_states = gen_confs(away - 1, save=save)
        for prev_action in prev_states:
            for prev_state_str in prev_states[prev_action]:
                for action in range(cube_env.action_space.n):
                    if is_inverse(action, prev_action):
                        continue
                    cube_env.set_state(str_to_state(prev_state_str))
                    cube_env.step(action)
                    res[action].add(state_to_str(cube_env.getstate()))

    if save:
        p = Path(f'{path}_{away}.pkl')
        with open(p, 'wb') as f:
            pickle.dump(res, f)
    return res
