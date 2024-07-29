import typing as t
import os
import subprocess
from os import popen

def system(cmd: str):
    cmd = cmd.strip().replace('\n', '')
    print('> ' + cmd)
    ret = os.system(cmd)
    if ret != 0: raise ChildProcessError(ret)


class SystemMemory(t.TypedDict):
    total_mb: int
    used_mb: int
    free_mb: int

def get_memory() -> SystemMemory:
    mem = [ int(col) for col in popen('free -t -m').readlines()[-1].split()[1:] ]
    return {
        'total_mb' : mem[0],
        'used_mb'  : mem[1],
        'free_mb'  : mem[2],
    }