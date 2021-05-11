from cube import CubeWrapper, actionList
import numpy as np

from collections import defaultdict
import pickle
from pathlib import Path
from tqdm import tqdm


def is_inverse(action1, action2):
    # action1, action2 = int(action1), int(action2)
    if actionList[action1] == '.' + actionList[action2] or '.' + actionList[action1] == actionList[action2]:
        return True
    return False


def str_to_state(state_str):
    return list(map(int, state_str.split()))


def state_to_str(state, one_hot=False):
    if one_hot:
        state = np.argmax(state, axis=1)
    return ' '.join(map(str, state))


def gen_confs(away=1, path='away', state_set=None, save=True):
    if state_set is None:
        state_set = set()
    res = set()
    cube_env = CubeWrapper(one_hot_states=False)
    cube_env.setScramble(1, 1, False)
    cube_env.reset()
    state_set.add(state_to_str(cube_env.state))
    if away == 1:
        for action in range(cube_env.action_space.n):
            cube_env.reset()
            cube_env.step(action)
            state_str = state_to_str(cube_env.state)
            state_set.add(state_str)
            res.add(state_str)
    else:
        pl = Path(f'{path}_{away - 1}.pkl')
        if pl.exists():
            with open(pl, 'rb') as f:
                prev_states = pickle.load(f)
            state_set = state_set.union(prev_states)
            for i in range(1, away - 1):
                p = pl.parent / Path(f'away_{i}.pkl')
                with open(p, 'rb') as f:
                    state_set = state_set.union(pickle.load(f))
        else:
            prev_states = gen_confs(away - 1, state_set=state_set, path=path, save=save)
        pbar = tqdm(total=len(prev_states) * 12)
        for prev_state_str in prev_states:
            for action in range(cube_env.action_space.n):
                cube_env.set_state(str_to_state(prev_state_str))
                cube_env.step(action)
                state_str = state_to_str(cube_env.state)
                if state_str not in state_set:
                    state_set.add(state_str)
                    res.add(state_str)
                pbar.update(1)
    # res = defaultdict(set)
    # cube_env = CubeWrapper(one_hot_states=False)
    # cube_env.setScramble(1, 2, doScamble=False)
    # if away == 1:
    #     for action in range(cube_env.action_space.n):
    #         cube_env.reset()
    #         cube_env.step(action)
    #
    #         res[action].add(state_to_str(cube_env.getstate()))
    # else:
    #     p = Path(f'{path}_{away - 1}.pkl')
    #     if p.exists():
    #         with open(p, 'rb') as f:
    #             prev_states = pickle.load(f)
    #     else:
    #         prev_states = gen_confs(away - 1, save=save)
    #     for prev_action in prev_states:
    #         for prev_state_str in prev_states[prev_action]:
    #             for action in range(cube_env.action_space.n):
    #                 if is_inverse(action, prev_action):
    #                     continue
    #                 cube_env.set_state(str_to_state(prev_state_str))
    #                 cube_env.step(action)
    #                 res[action].add(state_to_str(cube_env.getstate()))

    if save:
        p = Path(f'{path}_{away}.pkl')
        with open(p, 'wb') as f:
            pickle.dump(res, f)
    return res
