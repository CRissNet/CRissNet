import argparse  #

parser = argparse.ArgumentParser(description='TSnet PyTorch Training')

# ========================== Indispensable arguments ==========================

parser.add_argument('--data-dir', type=str, required=True,
                    help='the path of dataset.')
parser.add_argument('--scenario', type=str, required=True, choices=["in", "out"],
                    help="the channel scenario")
parser.add_argument('-b', '--batch-size', type=int, required=True, metavar='N',
                    help='mini-batch size')
parser.add_argument('--condition', type=str, required=True,
                    choices=["4i", "4o", "8i", "8o", "16i", "16o", "32i", "32o", "64i", "64o", "A", "B", "C", "D", "E"],
                    help="the channel scenario")
parser.add_argument('-j', '--workers', type=int, metavar='N', required=True,
                    help='number of data loading workers')  #

# ============================= Optical arguments =============================
#
# Working mode arguments
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', type=str, default=None,
                    help='using locally pre-trained model. The path of pre-trained model should be given')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--cpu', action='store_true',
                    help='disable GPU training (default: False)')
parser.add_argument('--cpu-affinity', default=None, type=str,
                    help='CPU affinity, like "0xffff"')  # default: 当参数需要默认值时，由这个参数指定

# Other arguments
parser.add_argument('--epochs', type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--scheduler', type=str, default='const', choices=['const', 'cosine'],
                    help='learning rate scheduler')
parser.add_argument('--n1', type=int, default='16', required=True,
                    help='CNN compress')
args = parser.parse_args()