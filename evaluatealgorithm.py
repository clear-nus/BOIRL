import numpy as np
import sys
import argparse
import boirlscenarios.constants as constants
from boirlscenarios.boirlmain import algosor
from boirlscenarios.calculatefetchsuccessrate import evalsuccessrate
np.random.seed(17870)

"""
MAIN SCRIPT TO ESOR OF THE GIVEN ALGORITHM IN THE SPECIFIED ENVIRONMENT
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
    algo = ""
    env = ""

    parser = argparse.ArgumentParser(description="Description for my parser")
    parser.add_argument("-a", "--algo", help="matern, rbf, taurbf, deepmaxent, airl or gcl", required=True, default="taurbf",
                        type=check_kernels)
    parser.add_argument("-e", "--env", help="gridworld2d, gridworld3d, vborlange or rborlange", required=True, default="gridworld2d",
                        type=check_envs)
    parser.add_argument("-p", "--projections", help="Size of projection space", type=int, required=False, default=10)
    argument = parser.parse_args()
    status = False

    if argument.algo:
        print("You have used '-a' or '--algo' with argument: {0}".format(argument.algo))
        algo = argument.algo
        status = True
    if argument.env:
        print("You have used '-e' or '--env' with argument: {0}".format(argument.env))
        env = argument.env
        status = True
    if argument.projections:
        print("You have used '-e' or '--env' with argument: {0}".format(argument.env))
        projs = argument.projections
        status = True

    if not status:
        print("Maybe you want to use -a and/or -e as arguments ?")

    #Calculate ESOR for the given algo in the specified env
    if env == constants.FETCH:
        evalsuccessrate(algo,env)
    else:
        algosor(algo, env, projs)



if __name__ == "__main__":
    main(sys.argv[1:])
