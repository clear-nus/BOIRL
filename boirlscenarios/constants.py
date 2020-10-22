#Environments
GRIDWORLD2D = "gridworld2d"
GRIDWORLD3D = "gridworld3d"
VIRTBORLANGE = "vborlange"
REALBORLANGE = "rborlange"
MAZE = "maze"
FETCH = "fetch"
allenvs = [GRIDWORLD2D,GRIDWORLD3D,VIRTBORLANGE,REALBORLANGE,MAZE,FETCH]

#Algorithms
RHORBF = "rhorbf"
RBF = "rbf"
MATERN = "matern"
AIRL = "airl"
GCL = "gcl"
BIRL = "birl"
trainablealgos = [RHORBF, RBF, MATERN, AIRL, GCL, BIRL]
allalgos = [RHORBF, RBF, MATERN, AIRL, GCL, BIRL]

#OpenAIGym
GRIDWORLDGYM = "SigmoidWorld-v0"
SWEDENWORLDGYM = "SwedenWorld-v0"
MAZEGYM = "PointMazeLeft-v0"

def isValidEnv(value):
    return value in allenvs

def isValidKernel(value):
    return value in trainablealgos

def get_gym(env):
    if env == GRIDWORLD3D:
        return GRIDWORLDGYM
    elif env == VIRTBORLANGE or env == REALBORLANGE:
        return SWEDENWORLDGYM
    elif env == MAZE:
        return MAZEGYM
    else:
        assert(NotImplementedError("OpenAI Gym implemented only for Gridworld3d and Borlange"))

#Plt Legends
LEGENDS={RHORBF: r"$\rho$" + "-RBF",
         RBF:"RBF",
         MATERN:"MATERN",
         AIRL:"AIRL",
         GCL:"GCL",
         BIRL:"BIRL"}