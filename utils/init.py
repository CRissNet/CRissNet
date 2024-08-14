import os
import random
import thop
import torch

from models import TSnet
from utils import logger, line_seg

__all__ = ["init_device", "init_model"]


def init_device(seed=None, cpu=None, gpu=None, affinity=None):
    # set the CPU affinity
    if affinity is not None:
        os.system(f'taskset -p {affinity} {os.getpid()}')  # system函数可以将字符串转化成命令在服务器上运行；其原理是每一条system
        # 函数执行时，其会创建一个子进程在系统上执行命令行，子进程的执行结果无法影响主进程  os.getpid()获取当前进程id

    # Set the random seed
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True  # 用来保证得到最好实验结果

    # Set the GPU id you choose
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)  # 可能有多个gpu，GPU进行编号并且在编号的GPU上运行

    # Env setup
    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')  # torch.device代表将torch.Tensor分配到的设备的对象
        torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(seed)
        pin_memory = True
        logger.info("Running on GPU%d" % (gpu if gpu else 0))
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info("Running on CPU")

    return device, pin_memory


def init_model(args):
    # Model loading
    model = TSnet(reduction=args.n1)

    if args.pretrained is not None:
        assert os.path.isfile(args.pretrained)
        state_dict = torch.load(args.pretrained,
                                map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(state_dict)
        logger.info("pretrained model loaded from {}".format(args.pretrained))

    # Model flops and params counting
    image = torch.randn([1, 2, 32, 32])
    flops, params = thop.profile(model, inputs=(image,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.3f")

    # Model info logging
    print(f'{line_seg}\n{model}\n{line_seg}\n')
    print(f'=> Model Name: TSnet [pretrained: {args.pretrained}]')
    print(f'=> Model Flops: {flops}')
    print(f'=> Model Params Num: {params}\n')

    return model