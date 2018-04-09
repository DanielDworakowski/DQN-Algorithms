import argparse
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('Configuration options.')
    parser.add_argument('--expName', dest='expName', default='', type=str, help='What to prefix names with on TB.')
    parser.add_argument('--config', dest='configStr', default='DefaultConfig', type=str, help='Name of the config file to import.')
    parser.add_argument('-n', dest='nConcurrent', default=1, help='Number of instances to launch.')
    args = parser.parse_args()
    return args
# 
# Launch multiple environments. 
def main()
    args = getInputArgs()
    for 
# 
# Launched as script.
if __name__ == "__main__":
    main()