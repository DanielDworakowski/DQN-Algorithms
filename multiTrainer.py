import argparse
import subprocess
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('Configuration options.')
    parser.add_argument('--expName', dest='expName', default='default', type=str, help='What to prefix names with on TB.')
    parser.add_argument('--config', dest='configStr', default='DefaultConfig', type=str, help='Name of the config file to import.')
    parser.add_argument('-n', dest='nConcurrent', default=1, type=int, help='Number of instances to launch.')
    parser.add_argument('--seed', dest='seedBase', default=1, type=int, help='The base number to use for the seed.')
    args = parser.parse_args()
    return args
# 
# Launch multiple environments. 
def main():
    args = getInputArgs()
    cfg = args.configStr
    exp = args.expName
    n = args.nConcurrent
    procs = []
    for tp in range(n):
        # 
        # Seed cannot be 0
        seed = args.seedBase + tp
        tmuxBeginCmd = 'tmux new -d -s %s%d '%(args.expName,seed)
        traincmd = '\'python train.py --config %s --seed %d --expName %s --useTB\''%(cfg, seed, exp) 
        # cmdList = tmuxBeginCmd.split()
        # cmdList.append(traincmd)
        # print(' '.join(cmdList))
        cmd = tmuxBeginCmd + traincmd
        procs.append(subprocess.Popen(cmd, shell=True))
    for proc in procs:
        proc.communicate()

# 
# Launched as script.
if __name__ == "__main__":
    main()