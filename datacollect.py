import numpy as np
import sys
import argparse
import boirlscenarios.constants as constants
from boirlscenarios.configurations import Configurations
import os
from boirlscenarios.irlobject import IRLObject

"""
MAIN SCRIPT TO COLLECT EXPERT TRAJECTORIES
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
    env = ""

    parser = argparse.ArgumentParser(description="Description for my parser")
    parser.add_argument("-e", "--env", help="gridworld2d, gridworld3d, vborlange or rborlange", required=True,
                        default="gridworld2d",
                        type=check_envs)
    argument = parser.parse_args()
    status = False

    if argument.env:
        print("You have used '-e' or '--env' with argument: {0}".format(argument.env))
        env = argument.env
        status = True

    if not status:
        print("Maybe you want to use -e as argument?")

    #Create new trajectories for all environments except fetch and maze
    # Fetch and Maze requires training of an expert policy to generate expert trajectories.
    # So we simply load a pre-existing trajectories
    if not (env == constants.FETCH or env == constants.MAZE):
        # Create an environment object
        configs = Configurations(None, env)
        if env == constants.GRIDWORLD2D:
            from mdp.gridworld2d import GridWorld2D
            O = GridWorld2D(horizon=configs.getLTrajs())
        elif env == constants.GRIDWORLD3D:
            from mdp.gridworld3d import GridWorld3D
            O = GridWorld3D(horizon=configs.getLTrajs())
        elif env == constants.VIRTBORLANGE or env == constants.REALBORLANGE:
            from mdp.borlangeworld import BorlangeWorld
            O = BorlangeWorld(destination=7622, horizon=configs.getLTrajs(), discount=configs.getDiscounts(),
                              loadres=True)

        # Get full trajectories
        if env == constants.REALBORLANGE:
            fullTrajs, fullSpos = O.gather_real_trajectories()
        elif env == constants.VIRTBORLANGE:
            print("Warning! This might take a while")
            fullTrajs, fullSpos, _ = O.generate_trajectories(n_trajectories=O.nodummy_states.shape[0],
                                                             startpos=np.random.permutation(
                                                                 np.random.permutation(O.nodummy_states)))
        elif env == constants.GRIDWORLD3D or env == constants.GRIDWORLD2D:
            fullTrajs, fullSpos, _ = O.generate_trajectories(n_trajectories=50, startpos=np.random.randint(0, 6, 50))
        else:
            raise AssertionError("Invalid Environment: %s") % env

        # Get trajectories for creating latent space
        n_full_trajectories, l_trajectories, _ = np.shape(fullTrajs)
        indices = np.random.permutation(np.arange(n_full_trajectories))[0:configs.getNTrajs()]
        trajectories = fullTrajs[indices]
        start_pos = fullSpos[indices]

        # Save them all in the Data directory
        np.save(os.path.join(configs.getTrajectoryDir(), "full_opt_trajectories.npy"), fullTrajs)
        np.save(os.path.join(configs.getTrajectoryDir(), "full_start_pos.npy"), fullSpos)
        np.save(os.path.join(configs.getTrajectoryDir(), "train_trajectories.npy"), trajectories)
        np.save(os.path.join(configs.getTrajectoryDir(), "feature_indices.npy"), indices)
        np.save(os.path.join(configs.getTrajectoryDir(), "features.npy"), O.features)

    else:
        print("We will reuse expert demonstrations stored in the Data folder for %s environment." % env)

    # Store reward function parameters within bounds to check rho-projection space
    if env == constants.GRIDWORLD2D:
        irlobj = IRLObject(None, env)
        W1 = np.linspace(irlobj.bounds[0]['domain'][0], irlobj.bounds[0]['domain'][1], 500)
        W2 = np.linspace(irlobj.bounds[1]['domain'][0], irlobj.bounds[1]['domain'][1], 500)
        w1, w2 = np.meshgrid(W1, W2)
        allw = np.hstack((w1.reshape(500 * 500, 1), w2.reshape(500 * 500, 1)))
        allw = np.append(allw, irlobj.gtheta, axis=0)
        allw = np.append(allw, -1 * irlobj.gtheta, axis=0)
        np.save(os.path.join(irlobj.configurations.getTrajectoryDir(), "allw.npy"), allw)
    elif env == constants.GRIDWORLD3D:
        irlobj = IRLObject(None, env)
        W1 = np.linspace(irlobj.bounds[0]['domain'][0], irlobj.bounds[0]['domain'][1], 500)
        W2 = np.linspace(irlobj.bounds[1]['domain'][0], irlobj.bounds[1]['domain'][1], 500)
        w1, w2 = np.meshgrid(W1, W2)
        tempw = np.hstack((w1.reshape(500 * 500, 1), w2.reshape(500 * 500, 1),np.zeros((500*500,1))))
        tempw = np.append(tempw, irlobj.gtheta, axis=0)
        tempw = np.append(tempw, -1 * irlobj.gtheta, axis=0)
        allw = np.vstack((tempw, tempw))
        allw = np.vstack((allw, tempw))
        allw[int(allw.shape[0] / 3):2 * int(allw.shape[0] / 3), 2] = -1
        allw[2 * int(allw.shape[0] / 3):, 2] = 2
        np.save(os.path.join(irlobj.configurations.getTrajectoryDir(), "allw.npy"), allw)
    elif env == constants.VIRTBORLANGE or env == constants.REALBORLANGE:
        irlobj = IRLObject(None, env)
        W1 = np.linspace(irlobj.bounds[0]['domain'][0], irlobj.bounds[0]['domain'][1], 100)
        W2 = np.linspace(irlobj.bounds[1]['domain'][0], irlobj.bounds[1]['domain'][1], 100)
        W3 = np.linspace(irlobj.bounds[2]['domain'][0], irlobj.bounds[2]['domain'][1], 100)
        w1, w2, w3 = np.meshgrid(W1, W2, W3)
        allw = np.hstack(
            (w1.reshape(100 * 100 * 100, 1), w2.reshape(100 * 100 * 100, 1), w3.reshape(100 * 100 * 100, 1)))
        if env == constants.VIRTBORLANGE:
            allw = np.append(allw, irlobj.gtheta, axis=0)
        np.save(os.path.join(irlobj.configurations.getTrajectoryDir(), "allw.npy"), allw)
    elif env == constants.MAZE or env == constants.FETCH:
        irlobj = IRLObject(None, env)
        W1 = np.linspace(irlobj.bounds[0]['domain'][0], irlobj.bounds[0]['domain'][1], 500)
        W2 = np.linspace(irlobj.bounds[1]['domain'][0], irlobj.bounds[1]['domain'][1], 500)
        w1, w2 = np.meshgrid(W1, W2)
        allw = np.hstack((w1.reshape(500 * 500, 1), w2.reshape(500 * 500, 1)))
        allw = np.append(allw, irlobj.gtheta, axis=0)
        np.save(os.path.join(irlobj.configurations.getTrajectoryDir(), "allw.npy"), allw)
    else:
        raise AssertionError("Invalid Environment: %s") % env


if __name__ == "__main__":
    main(sys.argv[1:])
