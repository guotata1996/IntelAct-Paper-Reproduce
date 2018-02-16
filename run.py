import subprocess
from config import config

from intelAct_optimizer import intelAct_masterprocess

if config.agent_num > 0:
    master = intelAct_masterprocess()
    master.start()

sub_proc = []
for k in range(config.local_agent_start, config.local_agent_end):
    print(k)
    p = subprocess.Popen('python -c "from client import ClientProcess; ClientProcess({}).start()"'.format(k))
    sub_proc.append(p)

for p in sub_proc:
    p.wait()
