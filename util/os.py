import os
import subprocess

def system(cmd: str):
    cmd = cmd.strip().replace('\n', '')
    print('> ' + cmd)
    ret = os.system(cmd)
    if ret != 0: raise ChildProcessError(ret)


def current_git_version() -> str:
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD']
    ).decode().strip()