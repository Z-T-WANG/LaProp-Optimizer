import torch
import torch.optim as optim
import os, glob, random
import numpy as np
from common.utils import load_model
from model import DQN

def test(env, args): 
    current_model = DQN(env, args).to(args.device)
    #current_model.eval()

    parse_key = args.load_model
    files = glob.glob('./models/*_*.pth')
    files = [name.split('/')[-1] for name in files]
    files = [name for name in files if name.split('_')[0] == parse_key]

    results = []
    for f in files:
        args.load_model = f
        num = int(f.split('_')[1].split('.')[0])
        performances = np.array([test_whole(current_model, env, args, num) for i in range(20)])
        results.append((num, np.mean(performances), np.std(performances, ddof=1)/np.sqrt(len(performances))))
        print(results[-1])

    results.sort(key=lambda result: result[0])

    with open('{}.txt'.format(parse_key), 'a') as data_f:
        for result in results:
            data_f.write('{}\t{}\t{}\n'.format(result[0], result[1], result[2]))


def test_whole(current_model, env, args, num):
    load_model(current_model, args)
    episode_reward = 0
    episode_length = 0

    state = env.reset()
    lives = env.unwrapped.ale.lives()
    live = lives
    while live > 0:
        for i in range(5000):
            if args.render:
                env.render()
            if args.noisy:
                current_model.update_noisy_modules()

            action = current_model.act(torch.FloatTensor(state).to(args.device), 0.)

            next_state, reward, done, _ = env.step(action)

            state = next_state
            episode_reward += reward
            episode_length += 1
            if done:
                state = env.reset()
                live -= 1
                break

        if not done: 
            while not done:
                if args.render:
                    env.render()
                _, _, done, _ = env.step(random.randrange(env.action_space.n))
            state = env.reset()
            live -= 1

    
    print("Test Result - Reward {} Length {} at {}".format(episode_reward, episode_length, num))
    return episode_reward
    
