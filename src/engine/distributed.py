import argparse
import os
import logging
import socket
import datetime

import torch


def is_global_master(args: argparse.Namespace) -> bool:
    return args.rank == 0


def is_local_master(args: argparse.Namespace) -> bool:
    return args.local_rank == 0


def is_master(args: argparse.Namespace, local: bool = False) -> bool:
    return is_local_master(args) if local else is_global_master(args)


def is_using_distributed() -> bool:
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    return False


def world_info_from_env() -> tuple[int, int, int]:
    local_rank, rank, world_size = 0, 0, 1
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])

    return local_rank, rank, world_size


def init_distributed_device(args: argparse.Namespace):
    assert args.device_mode in ('cuda', 'cpu'), f'{args.device_mode=} not supported'
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    if is_using_distributed():
        # DDP via torchrun, torch.distributed.launch
        args.local_rank, args.rank, args.world_size = world_info_from_env()
        # find new available port
        if not _is_free_port(os.environ["MASTER_PORT"]) and is_master(args):
            new_port = _find_free_port()
            os.environ["MASTER_PORT"] = str(new_port)
            print(f'find {new_port=}')
        if args.dist_backend == 'nccl':
            os.environ["NCCL_BLOCKING_WAIT"] = '1'
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            timeout=datetime.timedelta(hours=24)
        )
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        torch.distributed.barrier()
        args.distributed = True

    if args.device_mode == 'cuda' and torch.cuda.is_available():
        if args.distributed and not args.no_set_device_rank:
            device = f'cuda:{args.local_rank}'
        else:
            device = 'cuda:0'
        torch.cuda.set_device(device)
    elif args.device_mode == 'cpu':
        device = 'cpu'
    args.device = device


    if is_using_distributed():
        if is_master(args):
            print(f'Distributed mode enabled. {args.world_size=}')
    else:
        print('Not using distributed mode.')


def setup_print_for_distributed(args: argparse.Namespace):
    import builtins
    builtin_print = builtins.print

    def master_only_print(*print_args, **print_kwargs):
        force = print_kwargs.pop("force", False)
        if is_master(args) or force:
            builtin_print(*print_args, **print_kwargs)

    builtins.print = master_only_print


def _is_free_port(port: str | int) -> bool:
    port = int(port)
    ips = socket.gethostbyname_ex(socket.gethostname())[-1]
    ips.append("localhost")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return all(s.connect_ex((ip, port)) != 0 for ip in ips)


def _find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return int(port)
