import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='DQN')

    # Basic Arguments
    parser.add_argument('--seed', type=int, default=1122,
                        help='Random seed')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # Training Arguments
    parser.add_argument('--max-frames', type=int, default=1400000, metavar='STEPS',
                        help='Number of frames to train')
    parser.add_argument('--buffer-size', type=int, default=100000, metavar='CAPACITY',
                        help='Maximum memory buffer size')
    parser.add_argument('--update-target', type=int, default=1000, metavar='STEPS',
                        help='Interval of target network update')
    parser.add_argument('--train-freq', type=int, default=1, metavar='STEPS',
                        help='Number of steps between optimization step')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='γ',
                        help='Discount factor')
    parser.add_argument('--learning-start', type=int, default=10000, metavar='N',
                        help='How many steps of the model to collect transitions for before learning starts')
    parser.add_argument('--eps_start', type=float, default=1.0,
                        help='Start value of epsilon')
    parser.add_argument('--eps_final', type=float, default=0.01,
                        help='Final value of epsilon')
    parser.add_argument('--eps_decay', type=int, default=30000,
                        help='Adjustment parameter for epsilon')

    # Algorithm Arguments
    parser.add_argument('--double', action='store_true',
                        help='Enable Double-Q Learning')
    parser.add_argument('--dueling', action='store_true',
                        help='Enable Dueling Network')
    parser.add_argument('--noisy', action='store_true',
                        help='Enable Noisy Network')
    parser.add_argument('--prioritized-replay', action='store_true',
                        help='enable prioritized experience replay')
    parser.add_argument('--c51', action='store_true',
                        help='enable categorical dqn')
    parser.add_argument('--multi-step', type=int, default=1,
                        help='N-Step Learning')
    parser.add_argument('--Vmin', type=int, default=-10,
                        help='Minimum value of support for c51')
    parser.add_argument('--Vmax', type=int, default=10,
                        help='Maximum value of support for c51')
    parser.add_argument('--num-atoms', type=int, default=51,
                        help='Number of atom for c51')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha value for prioritized replay')
    parser.add_argument('--beta-start', type=float, default=0.4,
                        help='Start value of beta for prioritized replay')
    parser.add_argument('--beta-frames', type=int, default=100000,
                        help='End frame of beta schedule for prioritized replay')
    parser.add_argument('--sigma-init', type=float, default=0.4,
                        help='Sigma initialization value for NoisyNet')

    # Environment Arguments
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4',
                        help='Environment Name')
    parser.add_argument('--episode-life', type=int, default=1,
                        help='Whether env has episode life(1) or not(0)')
    parser.add_argument('--clip-rewards', type=int, default=1,
                        help='Whether env clip rewards(1) or not(0)')
    parser.add_argument('--frame-stack', type=int, default=1,
                        help='Whether env stacks frame(1) or not(0)')
    parser.add_argument('--scale', type=int, default=0,
                        help='Whether env scales(1) or not(0)')

    # Evaluation Arguments
    parser.add_argument('--load-model', type=str, default=None,
                        help='Pretrained model name to load (state dict)')
    parser.add_argument('--save-model', type=str, default='model',
                        help='Pretrained model name to save (state dict)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate only')
    parser.add_argument('--render', action='store_true',
                        help='Render evaluation agent')
    parser.add_argument('--evaluation_interval', type=int, default=10000,
                        help='Frames for evaluation interval')

    # Optimization Arguments
    parser.add_argument('--lr', type=float, default=1e-4, metavar='η',
                        help='Learning rate')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optimizer')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4,
                        help='epsilon of adam optimizer')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='Which GPU to use')
    parser.add_argument('--beta2', type=float, default=0.999)


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:{}".format(args.gpu_id) if args.cuda else "cpu")

    return args
