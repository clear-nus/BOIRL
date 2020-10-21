import numpy as np
import sys
import argparse
import boirlscenarios.constants as constants
from boirlscenarios.boirlmain import algoplot
from boirlscenarios.plotfetchsuccessrate import plot_fetch_sr
np.random.seed(168938)

"""
MAIN SCRIPT TO PLOT ESOR, NLL and also output the number of iteration to reach a certain percentage of expert's ESOR.
"""
def check_kernels(value):
    if not constants.isValidKernel(value):
        raise argparse.ArgumentTypeError("%s is an invalid kernel" % value)
    return value

def check_envs(value):
    if not constants.isValidEnv(value):
        raise argparse.ArgumentTypeError("%s is an invalid env" % value)
    return value

def main(argv):
    algos = []
    env = ""
    niter = 100
    percent = 1.

    parser = argparse.ArgumentParser(description="Description for my parser")
    parser.add_argument("-a", "--algo", help="matern, rbf, taurbf, deepmaxent, airl or gcl", nargs='+', required=True, default="taurbf",
                        type=check_kernels)
    parser.add_argument("-e", "--env", help="gridworld2d, gridworld3d, vborlange or rborlange", required=True, default="gridworld2d",
                        type=check_envs)
    parser.add_argument("-p", "--percent", help="Percentage of Expert SOR to report [0-1]", required=False, default=1,
                        type=float)
    parser.add_argument("-n", "--niter", help="Number of iterations to plot", required=False, default=100,
                        type=int)

    argument = parser.parse_args()
    status = False

    if argument.algo:
        print("You have used '-a' or '--algo' with argument: {0}".format(argument.algo))
        algos = argument.algo
        status = True
    if argument.env:
        print("You have used '-e' or '--env' with argument: {0}".format(argument.env))
        env = argument.env
        status = True
    if argument.niter:
        print("You have used '-n' or '--niter' with argument: {0}".format(argument.niter))
        niter = argument.niter
        status = True

    if argument.percent:
        print("You have used '-p' or '--percent' with argument: {0}".format(argument.percent))
        percent = argument.percent
        status = True

    if not status:
        print("Maybe you want to use -k, -e, -n and/or -p as arguments ?")

    if env == constants.FETCH:
        plot_fetch_sr(algos)
    else:
        algoplot(algos, env, percent, niter)



if __name__ == "__main__":
    main(sys.argv[1:])
