from dataclasses import dataclass, replace
from io import TextIOWrapper
import os
import shlex
import subprocess
import datetime
from pathlib import Path
from enum import Enum, auto
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple
import warnings
import argparse
import torch

from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg, dump_cfg, assert_cfg

# add the root directory to the python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from graphgps.utils import my_create_loader
from experiment_utils.constants import Dataset, ExecMode, Method
from experiment_utils.command import Command, save_command_list, load_command_list
from experiment_utils.device_pool import DevicePool, ProcessWrapper

DATASET_TO_NODE_ENCODER = {
    Dataset.COCO: "COCONode",
    Dataset.PeptStruct: "Atom",
    Dataset.PeptFunc: "Atom",
    Dataset.Pascal: "VOCNode",
    Dataset.PCQM: "Atom",
    Dataset.Paris: "CityNetworkNode",
    Dataset.Shanghai: "CityNetworkNode",
    Dataset.LA: "CityNetworkNode",
    Dataset.London: "CityNetworkNode",
    Dataset.ShapeNet: "LinearNode",
    Dataset.S3DIS: "LinearNode",
}

DATASET_TO_EDGE_ENCODER = {
    Dataset.COCO: "COCOEdge",
    Dataset.PeptStruct: "Bond",
    Dataset.PeptFunc: "Bond",
    Dataset.Pascal: "VOCEdge",
    Dataset.PCQM: "Bond",
    Dataset.Paris: "DummyEdge",
    Dataset.Shanghai: "DummyEdge",
    Dataset.LA: "DummyEdge",
    Dataset.London: "DummyEdge",
    Dataset.ShapeNet: "DummyEdge",
    Dataset.S3DIS: "DummyEdge",
}


def set_wandb_api_key():
    """
    Set the WANDB_API_KEY environment variable from the contents of 
    ~/jonas-fs/access_keys/WANDB_API_KEY.txt if it's not already set.
    Raises a warning if the file is not found.
    """
    # Skip if WANDB_API_KEY is already set
    if 'WANDB_API_KEY' in os.environ:
        print("Using the environment variable WANDB_API_KEY")
        return

    api_key_paths = [
        Path.home() / "jonas-fs" / "access_keys" / "WANDB_API_KEY.txt",
        Path(__file__).parent.parent.parent / "access_keys" / "WANDB_API_KEY.txt"
    ]

    api_key_path = next((path for path in api_key_paths if path.exists()), None)
    
    if not api_key_path:
        warnings.warn(f"WANDB API key file not found at any of the following locations: {api_key_paths}. WANDB logging will not work correctly.")
        return

    try:
        with open(api_key_path, 'r') as f:
            wandb_api_key = f.read().strip()
            os.environ['WANDB_API_KEY'] = wandb_api_key
            print(f"Successfully set WANDB_API_KEY from {api_key_path}")
    except Exception as e:
        warnings.warn(f"Failed to read WANDB API key from {api_key_path}: {str(e)}")


# Set WANDB API key when module is imported
set_wandb_api_key()



def preload_datasets(datasets: Optional[List[Dataset]] = None):
    if datasets is None:
        datasets = list(Dataset)
    print("Will preload the datasets: ", ", ".join(str(dataset) for dataset in datasets))
    for dataset in datasets:
        print(f"Preloading {dataset}...")
        cfg_path = f"configs/{dataset.benchmark}/{dataset}/defaults/GAT.yaml"
        if not Path(cfg_path).exists():
            raise FileNotFoundError(f"Config file does not exist: {cfg_path}")
        set_cfg(cfg)    # sets the default config values
        cfg.merge_from_file(cfg_path)
        assert_cfg(cfg)
        # custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)

        loaders = my_create_loader()


def run_configs_from_origin(cfg_origin: str, dataset: Dataset, method: str, results_dir: Path, exec_mode: ExecMode, start_seed=0, end_seed=0, cfg_file_check=True) -> List[Command]:
    cfg_path = f"configs/{dataset.benchmark}/{dataset}/{cfg_origin}/{method}.yaml"
    if cfg_file_check and not Path(cfg_path).exists():
        raise FileNotFoundError(f"Config file does not exist: {cfg_path}. Use --no-cfg_check to skip this check.")

    out_dir = results_dir / str(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    base_cmd = Command(
        cfg_path=cfg_path,
        out_dir=out_dir,
        seed=None,
        device_id=None,
        cfg_overrides="",
        exec_mode=exec_mode,
        cfg_origin=cfg_origin
    )

    commands = []
    for seed in range(start_seed, end_seed + 1):
        commands.append(replace(base_cmd, seed=seed))

    return commands


def linear_hp_search(dataset, method, results_dir: Path, exec_mode: ExecMode, start_seed=0, end_seed=0, cfg_file_check=True) -> List[Command]:
    cfg_path = f"configs/{dataset.benchmark}/{dataset}/defaults/{method}.yaml"
    if cfg_file_check and not Path(cfg_path).exists():
        raise FileNotFoundError(f"Config file does not exist: {cfg_path}. Use --no-cfg_check to skip this check.")

    out_dir = results_dir / str(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = Command(
        cfg_path=cfg_path,
        out_dir=out_dir,
        seed=None,
        device_id=None,
        cfg_overrides="",
        exec_mode=exec_mode
    )

    commands = []
    for seed in range(start_seed, end_seed + 1):
        # the default config
        commands.append(replace(base_cmd, seed=seed))

        # default: 0.1
        for dropout in [0.0, 0.2]:
            commands.append(replace(base_cmd, seed=seed, cfg_overrides=f"gnn.dropout {dropout} gt.dropout {dropout}"))

        # default: 8
        depths = [6,10] if dataset.benchmark != "CityNetworks" else [6,12,16]
        for depth in depths:
            commands.append(replace(base_cmd, seed=seed, cfg_overrides=f"gnn.layers_mp {depth} gt.layers {depth}"))

        # default: 0.001
        for lr in [0.005, 0.0005]:
            commands.append(replace(base_cmd, seed=seed, cfg_overrides=f"optim.base_lr {lr}"))

        # default: 2
        for head_depth in [1, 3]:
            commands.append(replace(base_cmd, seed=seed, cfg_overrides=f"gnn.layers_post_mp {head_depth}"))

        # encoding: default = none
        # don't use LapPE or RWSE for CityNetworks, as those graphs are too large
        if not dataset.benchmark == "CityNetworks":
            # LapPE
            commands.append(replace(base_cmd, seed=seed, cfg_overrides=f"dataset.node_encoder_name {DATASET_TO_NODE_ENCODER[dataset]}+LapPE posenc_LapPE.enable True"))
            # RWSE
            commands.append(replace(base_cmd, seed=seed, cfg_overrides=f"dataset.node_encoder_name {DATASET_TO_NODE_ENCODER[dataset]}+RWSE posenc_RWSE.enable True"))

        if str(method).startswith("GPS+"):
            transformer_type = str(method).split("+")[1]

            # inner MPNN: default GatedGCN
            # TODO: implement and add
            # commands.append(replace(base_cmd, seed=seed, cfg_overrides=f"layer_type GCN+{transformer_type}"))

            # default: BatchNorm
            # TODO: implement and add
            # commands.append(replace(base_cmd, seed=seed, cfg_overrides=f"gt.batch_norm False gt.layer_norm True"))

        if method == Method.kmip:
            # default: 15
            for k in [5, 45]:
                commands.append(replace(base_cmd, seed=seed, cfg_overrides=f"gt.sparse.k {k}"))
            
    return commands

def get_datasets(args: argparse.Namespace) -> List[Dataset]:
    if not args.datasets:
        return list(Dataset)
    elif all(d in Dataset.keys() for d in args.datasets):
        return [Dataset[d] for d in args.datasets]
    elif len(args.datasets) == 1 and args.datasets[0] in Dataset.valid_identifiers():
        return Dataset.identifier_to_dataset_list(args.datasets[0])
    else:
        raise ValueError(f"Invalid dataset identifiers: {args.datasets}")

def get_methods(args: argparse.Namespace) -> List[Method]:
    if not args.methods:
        return list(Method)
    elif all(m in Method.keys() for m in args.methods):
        return [Method[m] for m in args.methods]
    elif len(args.methods) == 1 and args.methods[0] in Method.valid_identifiers():
        return Method.identifier_to_method_list(args.methods[0])
    else:
        raise ValueError(f"Invalid method identifiers: {args.methods}")


def gather_commands(args: argparse.Namespace, datasets: List[Dataset], methods: List[Method], exec_mode: ExecMode) -> List[Command]:
    if args.commands == "hp-search":
        commands = []
        for dataset in datasets:
            for method in methods:
                commands.extend(linear_hp_search(dataset, method, args.results_dir, exec_mode, cfg_file_check=not args.no_cfg_check, start_seed=args.start_seed, end_seed=args.end_seed))
        return commands

    elif args.commands in ["best-of-each", "tuned", "tuned-so-far", "defaults"]:
        commands = []
        for dataset in datasets:
            for method in methods:
                commands.extend(run_configs_from_origin(args.commands, dataset, method, args.results_dir, exec_mode, cfg_file_check=not args.no_cfg_check, start_seed=args.start_seed, end_seed=args.end_seed))
        return commands

    else:
        raise ValueError(f"Invalid command identifier: {args.commands}")


def gather_commands_from_file(args: argparse.Namespace) -> Tuple[List[Dataset], List[Method], List[Command]]:
    # assert command is an existing path
    try:
        path = Path(args.commands)
        commands = load_command_list(path)
        datasets = list(set(Dataset[cmd.dataset] for cmd in commands))
        methods = list(set(Method[cmd.method] for cmd in commands))
    except FileNotFoundError:
        raise FileNotFoundError(f"Commands file does not exist: {args.commands}")
    except Exception as e:
        raise Exception(f"Error loading commands from {args.commands}: {e}")
    return datasets, methods, commands


def run_commands(commands: Iterable[Command], device: str, n_gpus: Optional[int] = None, n_processes_per_device: int = 1):
    if n_gpus is None:
        n_gpus = torch.cuda.device_count()
    if n_gpus == 0 and device == "cuda":
        raise ValueError("No GPUs found")

    print(f"Running {len(list(commands))} commands on {n_gpus} GPUs with {n_processes_per_device} processes per GPU")
    
    device_pool = DevicePool(n_gpus, n_processes_per_device)
    command_queue = list(commands)

    try:
        while command_queue or device_pool.num_occupied > 0:
            # print current state of the device pool
            device_pool.pretty_print()

            device_pool.clear_inactive()
            
            # Start new processes if there's room in device pools
            while command_queue and device_pool.num_occupied < device_pool.capacity:
                # let device id be the device with the smallest number of processes
                device_id = device_pool.least_occupied_device()
                
                if len(device_pool.device_to_process[device_id]) < n_processes_per_device:
                    cmd = command_queue.pop(0)
                    cmd = replace(cmd, device_id=(device_id if device != "cpu" else "cpu"))
                    proc, stdout_file, stderr_file = cmd.execute(device)
                    if proc:  # Only track if a real process was started
                        process_wrapper = ProcessWrapper(proc, stdout_file, stderr_file, cmd)
                        device_pool.add(device_id, process_wrapper)
            
            # Wait before checking again
            time.sleep(3)

    except KeyboardInterrupt:
        print("Keyboard interrupt. Killing all processes...")
        device_pool.kill_all()
        raise

    except Exception as e:
        raise Exception(f"Error {e} occurred. Killing the mother process. If you want to kill the child processes, you should do so manually.")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run LRGB experiments')
    parser.add_argument('mode', choices=['print', 'test', 'real'], 
                        help='Mode to run: print (just print commands), test (run test configs), or real (execute commands)')
    parser.add_argument('--no-cfg-check', action='store_true', 
                        help='Skip config file check')
    parser.add_argument('--pretend-n-gpus', type=int, default=None,
                        help='Pretend to have this many GPUs')
    parser.add_argument('--cpu', action='store_true',
                        help='Force using CPU instead of CUDA')
    parser.add_argument('--datasets', nargs='+', choices=[d.name for d in Dataset] + Dataset.valid_identifiers(),
                        help=f'List of datasets to preload and train on. If not specified, all datasets will be processed. You can also use any of the following identifiers: {", ".join(Dataset.valid_identifiers())}')
    parser.add_argument('--methods', nargs='+', choices=[m.name for m in Method] + Method.valid_identifiers(),
                        help=f'List of methods to run. If not specified, all methods will be processed. You can also use any of the following identifiers: {", ".join(Method.valid_identifiers())}')
    parser.add_argument('--results-dir', type=Path, default=Path("results"),
                        help='Directory to save results')
    parser.add_argument('--commands', type=str, default="hp-search",
                        help='Either a mode ("hp-search", "best-of-each", "tuned", "tuned-so-far", "defaults") or a path to a JSON file containing commands to run')
    parser.add_argument('--save-commands', type=Path, default=None,
                        help='Path to a JSON file to save the commands to. If not specified, commands will not be saved.')
    parser.add_argument('--start-seed', type=int, default=0,
                        help='Starting seed for experiment runs (inclusive)')
    parser.add_argument('--end-seed', type=int, default=0,
                        help='Ending seed for experiment runs (inclusive)')
    parser.add_argument('--cfg_overrides', type=str, default="",
                        help='Additional config overrides as a string')
    args = parser.parse_args()
    
    # Define execution mode
    exec_mode = ExecMode(args.mode)
    device = "cpu" if args.cpu else "cuda"

    if args.results_dir == Path("results"):
        # Ensure the results directory exists
        args.results_dir.mkdir(parents=True, exist_ok=True)
    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {args.results_dir}")

    if args.commands in ["hp-search", "best-of-each", "tuned", "tuned-so-far", "defaults"]:
        datasets = get_datasets(args)
        methods = get_methods(args)
        commands = gather_commands(args, datasets, methods, exec_mode)
    else:
        datasets, methods, commands = gather_commands_from_file(args)
        commands = [replace(cmd, exec_mode=exec_mode) for cmd in commands]

    if args.cfg_overrides:
        commands = [replace(cmd, cfg_overrides=cmd.cfg_overrides + " " + args.cfg_overrides) for cmd in commands]
    
    if args.save_commands:
        print(f"Saving commands to {args.save_commands}")
        args.save_commands.parent.mkdir(parents=True, exist_ok=True)
        save_command_list(commands, args.save_commands)
    
    if exec_mode != ExecMode.PRINT:
        preload_datasets(datasets)

    run_commands(commands, device, n_gpus=args.pretend_n_gpus)


if __name__ == "__main__":
    main()