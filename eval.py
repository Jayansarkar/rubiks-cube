from cube import CubeWrapper
from train1 import load_confs, conf_and_automation_reward, Agent
from tqdm import trange

INPUT_SHAPE = (54, 6)

if __name__ == '__main__':
    env = CubeWrapper(max_step=100)
    env.setScramble(1, 30)
    main_agent = Agent(input_shape=INPUT_SHAPE)
    away_agents = [Agent(f'away_{i}', input_shape=INPUT_SHAPE) for i in range(1, 7)]
    agents = [main_agent] + away_agents
    # load agents
    for agent in agents:
        agent.load()
    print('Loading done')
    confs = load_confs()
    count = 0
    reward_arr = [None] * 500
    t = trange(500)
    for i in t:
        conf = conf_and_automation_reward(env.reset(), confs)
        active_agent = agents[conf]
        eval_reward = 0
        done = False
        while not done:
            current_state = env.getstate()
            action = active_agent.get_action(current_state, eval=True)
            new_state, reward, done, _ = env.step(action)

            if not done:
                conf, ar = conf_and_automation_reward(new_state, confs, last_conf=conf)
            else:
                ar = 1

            active_agent = agents[conf]

            reward = reward + ar
            eval_reward += reward
        reward_arr[i] = eval_reward
        if env.ncube.isSolved():
            count += 1
        t.set_description_str(f'count={count}/{i + 1}')
