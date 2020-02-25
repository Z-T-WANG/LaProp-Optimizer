import math
import os
import datetime
import time
import pathlib
import random

import torch
import numpy as np

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def epsilon_scheduler(eps_start, eps_final, eps_decay):
    def function(frame_idx):
        return eps_final + (eps_start - eps_final) * math.exp(-1. * frame_idx / eps_decay)
    return function

def beta_scheduler(beta_start, beta_frames):
    def function(frame_idx):
        return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
    return function

def create_log_dir(args):
    log_dir = ""
    if args.multi_step != 1:
        log_dir = log_dir + "{}-step-".format(args.multi_step)
    if args.c51:
        log_dir = log_dir + "c51-"
    if args.prioritized_replay:
        log_dir = log_dir + "per-"
    if args.dueling:
        log_dir = log_dir + "dueling-"
    if args.double:
        log_dir = log_dir + "double-"
    if args.noisy:
        log_dir = log_dir + "noisy-"
    log_dir = log_dir + "dqn-"
    
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = log_dir + now

    log_dir = os.path.join("runs", log_dir)
    return log_dir

def print_log(frame, prev_frame, prev_time, reward_list, length_list, loss_list, args):
    fps = (frame - prev_frame) / (time.time() - prev_time)
    avg_reward = np.mean(reward_list)
    avg_length = np.mean(length_list)
    avg_loss = np.mean(loss_list) if len(loss_list) != 0 else 0.

    print("Frame: {:<8} FPS: {:.2f} Avg. Reward: {:.2f} Avg. Length: {:.2f} Avg. Loss: {:.4f}".format(
        frame, fps, avg_reward, avg_length, avg_loss
    ))
    with open('data_{}.txt'.format(args.optim), 'a') as f:
        f.write('{:.0f}\t{}\n'.format((frame + prev_frame)/2., avg_reward))

def print_args(args):
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

def save_model(model, args, name=None):
    if name != None:
        torch.save(model.state_dict(), os.path.join("models", name) + ".pth")
        return

    fname = ""
    if args.multi_step != 1:
        fname += "{}-step-".format(args.multi_step)
    if args.c51:
        fname += "c51-"
    if args.prioritized_replay:
        fname += "per-"
    if args.dueling:
        fname += "dueling-"
    if args.double:
        fname += "double-"
    if args.noisy:
        fname += "noisy-"
    fname += "dqn-{}.pth".format(args.save_model)
    fname = os.path.join("models", fname)

    pathlib.Path('models').mkdir(exist_ok=True)
    torch.save(model.state_dict(), fname)

def load_model(model, args):
    if args.load_model is not None:
        fname = os.path.join("models", args.load_model)
    else:
        fname = ""
        if args.multi_step != 1:
            fname += "{}-step-".format(args.multi_step)
        if args.c51:
            fname += "c51-"
        if args.prioritized_replay:
            fname += "per-"
        if args.dueling:
            fname += "dueling-"
        if args.double:
            fname += "double-"
        if args.noisy:
            fname += "noisy-"
        fname += "dqn-{}.pth".format(args.save_model)
        fname = os.path.join("models", fname)

    if args.device == torch.device("cpu"):
        map_location = lambda storage, loc: storage
    else:
        map_location = None
    
    if not os.path.exists(fname):
        raise ValueError("No model saved with name {}".format(fname))

    model.load_state_dict(torch.load(fname, map_location))

def set_global_seeds(seed):
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    np.random.seed(seed)
    random.seed(seed)
