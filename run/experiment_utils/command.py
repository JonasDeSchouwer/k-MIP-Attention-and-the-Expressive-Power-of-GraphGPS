from dataclasses import dataclass, replace, asdict
from typing import List, Optional
import datetime
import subprocess
import json
from pathlib import Path
from enum import Enum

from .constants import ExecMode


@dataclass(frozen=True)
class Command:
    cfg_path: str
    out_dir: Path
    cfg_overrides: str
    seed: Optional[int]
    device_id: Optional[int]
    exec_mode: ExecMode
    cfg_origin: Optional[str] = None

    def __str__(self):
        return self.main_cmd
    
    @property
    def dataset(self) -> str:
        return Path(self.cfg_path).parent.parent.name
    
    @property
    def method(self):
        return Path(self.cfg_path).stem
    
    @property
    def exec_mode_overrides(self) -> str:
        """How the config should be modified to run in the given exec mode"""
        if self.exec_mode == ExecMode.PRINT:
            return ""
        elif self.exec_mode == ExecMode.TEST:
            return "optim.max_epoch 1 wandb.project tests"
        elif self.exec_mode == ExecMode.REAL:
            return ""
        else:
            raise NotImplementedError(f"Not yet implemented for exec mode: {self.exec_mode}")
        
    @property
    def total_cfg_overrides(self) -> str:
        """How the config will be overridden in the command"""
        return self.cfg_overrides + " " + self.exec_mode_overrides

    @property
    def cmd(self):
        result = "echo '=== Hardware Info ===' && "
        result += "nvidia-smi && "
        result += "lscpu && "
        result += "echo '=== Running Command ===' && "
        result += f"echo 'Command: conda run -n kmipattn {self.main_cmd}' && "

        result += "conda run -n kmipattn " + self.main_cmd

        return result
    
    @property
    def main_cmd(self):
        """The main part of the command, excluding clutter"""
        result = f"python -O main.py --cfg {self.cfg_path}"

        if self.seed is not None:
            result += f" --repeat 1 seed {self.seed}"

        result += f" out_dir {self.out_dir}"
        result += f" {self.total_cfg_overrides}"
        result += f" name_tag {self.name_tag}"

        if self.device_id is not None:
            result = f"CUDA_VISIBLE_DEVICES={self.device_id} {result} device cuda:0"

        return result
    
    @property
    def name_tag(self) -> str:
        if self.cfg_overrides:
            if self.cfg_origin:
                return f"{self.cfg_origin}-" + '-'.join(self.cfg_overrides.split())
            else:
                return '-'.join(self.cfg_overrides.split())
        else:
            if self.cfg_origin:
                return self.cfg_origin
            else:
                return "default"
    
    @property
    def out_dir_for_this_seed(self) -> Path:
        """The directory to save .out and .err files"""
        # return self.out_dir / f"{self.method}-{self.name_tag}" / "out-err" / str(self.seed)
        return self.out_dir.parent / "out-err" / self.dataset / f"{self.method}-{self.name_tag}" / str(self.seed)
    
    @property
    def stdout_path(self) -> Path:
        timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
        return self.out_dir_for_this_seed / f"{timestamp}-{self.method}-{self.seed}-{'-'.join(self.total_cfg_overrides.split())}.out"
    
    @property
    def stderr_path(self) -> Path:
        timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
        return self.out_dir_for_this_seed / f"{timestamp}-{self.method}-{self.seed}-{'-'.join(self.total_cfg_overrides.split())}.err"

    def to_json_string(self) -> str:
        """Convert the command to a single-line JSON string."""
        return json.dumps(asdict(self))
    
    @staticmethod
    def from_json_string(json_string: str) -> 'Command':
        """Convert a single-line JSON string to a Command object."""
        return Command(**json.loads(json_string))
    
    def execute(self, device: str) -> tuple[subprocess.Popen, Path, Path]:
        if self.exec_mode == ExecMode.PRINT:
            print(self)
            return None, None, None

        else:
            # ensure the parent dir of stdout_path and stderr_path exists
            self.stdout_path.parent.mkdir(parents=True, exist_ok=True)
            self.stderr_path.parent.mkdir(parents=True, exist_ok=True)

            stdout_file = open(self.stdout_path, "w")
            stderr_file = open(self.stderr_path, "w")

            if device == "cpu":
                cmd = replace(self, cfg_overrides=self.cfg_overrides + " device cpu")
            else:
                cmd = self
            
            return subprocess.Popen(cmd.cmd, stdout=stdout_file, stderr=stderr_file, shell=True), stdout_file, stderr_file

def custom_serialization(obj):
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def custom_deserialization(dct):
    for k, v in dct.items():
        if k == 'cfg_path':
            dct[k] = str(v)
        elif k == 'out_dir':
            dct[k] = Path(v)
        elif k == 'cfg_overrides':
            dct[k] = str(v)
        elif k == 'seed':
            dct[k] = int(v) if v is not None else None
        elif k == 'device_id':
            dct[k] = int(v) if v is not None else None
        elif k == 'exec_mode':
            dct[k] = ExecMode(v)
    return dct

def save_command_list(command_list: List[Command], json_path: Path):
    """Save a list of commands to a JSON file."""
    with open(json_path, 'w') as f:
        json.dump([asdict(cmd) for cmd in command_list], f, indent=2, default=custom_serialization)

def load_command_list(json_path: Path) -> List[Command]:
    """Load a list of commands from a JSON file."""
    with open(json_path, 'r') as f:
        return [Command(**cmd_dict) for cmd_dict in json.load(f, object_hook=custom_deserialization)]
    