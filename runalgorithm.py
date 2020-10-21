import sys
import argparse
from boirlscenarios.boirlmain import algoexecute
import boirlscenarios.constants as constants
import numpy as np
from boirlscenarios.configurations import Configurations
import timeit
import os

np.random.seed(3543)
"""
MAIN SCRIPT TO RUN THE GIVEN ALGORITHM ON A SPECIFIC ENVIRONMENT
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
    budget = 100
    trials = 3
    nInits = 1

    parser = argparse.ArgumentParser(description="Description for my parser")
    parser.add_argument("-a", "--algo", help="matern, rbf, rhorbf, deepmaxent, airl or gcl", required=True, default="rhorbf",
                        type=check_kernels)
    parser.add_argument("-e", "--env", help="gridworld2d, gridworld3d, vborlange or rborlange", required=True, default="gridworld2d",
                        type=check_envs)
    parser.add_argument("-b", "--budget", help="Budget for BO", type=int, required=False, default=100)
    parser.add_argument("-n", "--trials", help="Number of times to repeat the experiment", type=int, required=False, default=3)
    parser.add_argument("-i", "--ninits", help="Number of random points for initialization", type=int, required=False, default=1)
    parser.add_argument("-p", "--projections", help="Size of projection space", type=int, required=False, default=None)
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
    if argument.budget:
        print("You have used '-b' or '--budget' with argument: {0}".format(argument.budget))
        budget = argument.budget
        status = True
    if argument.trials:
        print("You have used '-n' or '--trials' with argument: {0}".format(argument.trials))
        trials = argument.trials
        status = True
    if argument.ninits:
        print("You have used '-i' or '--ninits' with argument: {0}".format(argument.ninits))
        nInits = argument.ninits
        status = True
    if argument.projections:
        print("You have used '-p' or '--projections' with argument: {0}".format(argument.projections))
        projections = argument.projections
    else:
        projections = None

    if not status:
        print("Maybe you want to use -a, -e, -b and/or -n as arguments ?")

    # Run the given algorithm in the specified environment for "budget" number of iterations. Repeat this for "trials" time.
    config = Configurations(algo, env, projections)
    stime = timeit.default_timer() #Start a timer
    algoexecute(algo, env, budget, trials, nInits, projections)
    etime = (timeit.default_timer() - stime) / trials #end timer and store average execution time
    np.save(os.path.join(config.getResultDir(), "TimeTaken.npy"), etime)


if __name__ == "__main__":
    main(sys.argv[1:])
