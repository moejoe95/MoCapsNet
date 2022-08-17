import os
import torch
import torchvision.transforms as transforms
from trainer import CapsNetTrainer
import argparse
from small_norb import smallNORB


torch.manual_seed(23545)

DATA_PATH = '/home/josef.gugglberger/data'

# Collect arguments (if any)
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='MNIST',
                    help="'MNIST', 'SVHN' or 'CIFAR' (case insensitive).")
# Batch size
parser.add_argument('-bs', '--batch_size', type=int,
                    default=128, help='Batch size.')
# Epochs
parser.add_argument('-e', '--epochs', type=int,
                    default=30, help='Number of epochs.')
# Learning rate
parser.add_argument('-lr', '--learning_rate', type=float,
                    default=1e-3, help='Learning rate.')
# Number of routing iterations
parser.add_argument('--num_routing', type=int, default=3,
                    help='Number of routing iteration in routing capsules.')

# routing algorithm
parser.add_argument('--routing', type=str, default="RBA",
                    help='RBA or SDA routing.')

# Exponential learning rate decay
parser.add_argument('--lr_decay', type=float, default=0.96,
                    help='Exponential learning rate decay.')
# Select device "cuda" for GPU or "cpu"
parser.add_argument('--device', type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                    choices=['cuda', 'cpu'], help='Device to use. Choose "cuda" for GPU or "cpu".')
# Use multiple GPUs?
parser.add_argument('--multi_gpu', action='store_true',
                    help='Flag whether to use multiple GPUs.')
# Select GPU device
parser.add_argument('--gpu_device', type=int, default=None,
                    help='ID of a GPU to use when multiple GPUs are available.')
# Data directory
parser.add_argument('--data_path', type=str, default=DATA_PATH,
                    help='Path to the MNIST or CIFAR dataset. Alternatively you can set the path as an environmental variable $data.')

# use residual learning
parser.add_argument('-res', '--residual', dest='residual', action='store_true',
                    help='Use residual shortcut connections.')

# measure conflicting bundles
parser.add_argument('-cb', '--conflicts', dest='conflicts', action='store_true',
                    help='Measure conflicting bundles.')

# conflicting bundles batch size
parser.add_argument('-cb_bs', '--cb_batch_size', type=int,
                    default=32, help='Batch size of conflicting bundles.')

# use momentum
parser.add_argument('-m', '--momentum', dest='momentum', action='store_true',
                    help='Use residual shortcut connections..')
parser.add_argument('-g', '--gamma', type=float,
                    default=0.9, help='Momentum term.')

parser.add_argument('-b', '--num_res_blocks', type=int,
                    default=1, help='Number of residual blocks.')

parser.add_argument('-c', '--num_caps', type=int,
                    default=32, help='Number of capsules.')

# optimizer
parser.add_argument('-o', '--optimizer', type=str,
                    default='adam', help='One of: ranger21, adam')

args = parser.parse_args()

if not args.residual and not args.momentum:
    args.modelname = "CapsNet_" + str(args.num_res_blocks)
elif args.residual and not args.momentum:
    args.modelname = "ResCapsNet_" + str(args.num_res_blocks)
elif args.residual and args.momentum:
    args.modelname = "MoCapsNet_" + str(args.num_res_blocks)

device = torch.device(args.device)

if args.gpu_device is not None:
    torch.cuda.set_device(args.gpu_device)

if args.multi_gpu:
    args.batch_size *= torch.cuda.device_count()


classes = ['animal', 'human', 'airplane', 'truck', 'car']
args.num_classes = len(classes)

transform_train = transforms.Compose([
    transforms.Resize(48),
    transforms.RandomCrop(32),
    transforms.ColorJitter(brightness=32./255, contrast=0.5),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.Resize(48),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])

loaders = {}

loaders['train'] = torch.utils.data.DataLoader(
    smallNORB(DATA_PATH, train=True, download=True, transform=transform_train), batch_size=args.batch_size, shuffle=True)

loaders['test'] = torch.utils.data.DataLoader(
    smallNORB(DATA_PATH, train=False, transform=transform_test), batch_size=args.batch_size, shuffle=True)


print(8*'#', f'Using {args.dataset.upper()} dataset', 8*'#')

# Run
caps_net = CapsNetTrainer(loaders, args, device=device)
caps_net.run(args.epochs, classes=classes)
