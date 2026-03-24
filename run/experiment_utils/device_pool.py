from io import TextIOWrapper
import subprocess
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List

from .command import Command

@dataclass
class ProcessWrapper:
    proc: subprocess.Popen
    stdout_file: TextIOWrapper
    stderr_file: TextIOWrapper
    command: Command

    @property
    def finished(self):
        return self.proc.poll() is not None

    def close(self) -> int:
        """
        Close the process and return the return code.
        """
        self.stdout_file.close()
        self.stderr_file.close()
        
        # check return code
        if self.proc.returncode != 0:
            warnings.warn(f"Process {self.command.method} on {self.command.dataset} with seed {self.command.seed} returned non-zero exit code {self.proc.returncode}")

        return self.proc.returncode


class DevicePool:
    def __init__(self, n_gpus: int, n_processes_per_device: int):
        self.n_gpus = n_gpus
        self.n_processes_per_device = n_processes_per_device
        self.device_to_process: Dict[int, List[ProcessWrapper]] = {i: [] for i in range(n_gpus)}
        self.device_stats: Dict[int, Dict[str, int]] = {i: {"succeeded": 0, "failed": 0} for i in range(n_gpus)}
        self.failed_commands: List[Command] = []

    def __len__(self):
        return sum(len(pool) for pool in self.device_to_process.values())
    
    def keys(self):
        return self.device_to_process.keys()
    
    def values(self):
        return self.device_to_process.values()
    
    def items(self):
        return self.device_to_process.items()
    
    def pretty_print(self):
        """
        Print a table that contains, for each device in the pool, a summary of each process running on that device.
        The summary includes cfg_path, cfg_overrides, and seed of each command.
        """
        lines = self._pretty_print_lines()
        for line in lines:
            print(line)

    def _pretty_print_lines(self) -> List[str]:
        """
        Return the lines of a table that contains, for each device in the pool, a summary of each process running on that device.
        The summary includes cfg_path, cfg_overrides, and seed of each command.
        """
        
        lines = [
            "\n" + "=" * 100,
            f"Device Pool Status ({self.num_occupied}/{self.capacity} processes running)",
            "=" * 100,
        ]
        lines[1] += str(time.strftime('%H:%M:%S')).rjust(100-len(lines[1]))
        
        for device_id, processes in self.device_to_process.items():                
            lines.append(f"Device {device_id}: Running {len(processes)}/{self.n_processes_per_device} - Failed {self.device_stats[device_id]['failed']} - Succeeded {self.device_stats[device_id]['succeeded']}")

            if processes:
                lines.extend([
                    "-" * 100,
                    f"{'Process Status':<15} {'Config Path':<40} {'Seed':<8} {'Config Overrides'}",
                    "-" * 100,
                ])
            
            for proc_wrapper in processes:
                status = "Running" if proc_wrapper.proc.poll() is None else "Finished"
                cmd = proc_wrapper.command
                cfg_path = "/".join(cmd.cfg_path.split("/")[-3:]) if cmd.cfg_path else "N/A"
                seed = cmd.seed if cmd.seed is not None else "N/A"
                cfg_overrides = cmd.cfg_overrides if cmd.cfg_overrides else "None"
                
                lines.append(f"{status:<15} {cfg_path:<40} {seed:<8} {cfg_overrides}")
            
            lines.append("=" * 100)

        if self.failed_commands:
            lines.append("=" * 100)
            lines.append("Failed Commands:")
            for cmd in self.failed_commands:
                lines.append(f"{cmd.main_cmd}")
            lines.append("=" * 100)

        return lines
    
    @property
    def num_occupied(self):
        return sum(len(pool) for pool in self.device_to_process.values())
    
    @property
    def capacity(self):
        return self.n_gpus * self.n_processes_per_device
    
    def add(self, device_id: int, proc: ProcessWrapper):
        assert device_id == proc.command.device_id
        self.device_to_process[device_id].append(proc)
    
    def clear_inactive(self):
        for device_id, processes in list(self.device_to_process.items()):
            for process_wrapper in processes:
                if process_wrapper.finished:
                    return_code = process_wrapper.close()
                    self.device_to_process[device_id].remove(process_wrapper)
                    if return_code == 0:
                        self.device_stats[device_id]["succeeded"] += 1
                    else:
                        self.device_stats[device_id]["failed"] += 1
                        self.failed_commands.append(process_wrapper.command)

    def least_occupied_device(self):
        return min(self.device_to_process.keys(), key=lambda x: len(self.device_to_process[x]))
    
    def kill_all(self):
        for device_id, processes in self.device_to_process.items():
            for process_wrapper in processes:
                process_wrapper.proc.kill()
                return_code = process_wrapper.close()
                if return_code == 0:
                    self.device_stats[device_id]["succeeded"] += 1
                else:
                    self.device_stats[device_id]["failed"] += 1
                    self.failed_commands.append(process_wrapper.command)
