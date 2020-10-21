import numpy as np
from boirlscenarios.irlobject import IRLObject
from boirlscenarios.configurations import Configurations
from tqdm import tqdm
import GPyOpt
import os
import boirlscenarios.constants as constants
import matplotlib.pyplot as plt
from tabulate import tabulate
import time


def exp_moving_average(data, alpha=0.6):
    smooth_data = np.zeros(np.shape(data))
    for dind, d in enumerate(data):
        prev_sv = None
        for lind, l in enumerate(d):
            if lind == 0:
                sv = l
            else:
                sv = alpha * l + (1 - alpha) * prev_sv
            smooth_data[dind, lind] = sv
            prev_sv = sv
    return smooth_data


def get_experts_fetch(O):
    import gym
    KEY_ORDER = ['observation', 'achieved_goal', 'desired_goal']
    fulltrajectories = O.trajectories
    indices = fulltrajectories["indices"]
    inds = np.where(indices == 1.)[0]
    inds = np.append(inds, indices.shape[0]).astype(np.int)
    expert_trajs = []
    for n in range(len(inds) - 1):
        observation_dic = None
        for k in KEY_ORDER:
            if observation_dic is None:
                observation_dic = fulltrajectories[k][inds[n]:inds[n + 1]]
            else:
                observation_dic = np.append(observation_dic, fulltrajectories[k][inds[n]:inds[n + 1]], axis=1)

        action = fulltrajectories["actions"][inds[n]:inds[n + 1]]
        fin_dic = {"observations": observation_dic,
                   "actions": action}
        expert_trajs.append(fin_dic)
    return expert_trajs


def get_experts(dir, trajectories=None):
    all_obs = np.expand_dims(np.load(os.path.join(dir, "features.npy")), axis=1)
    if len(np.shape(all_obs)) > 2:
        all_obs = np.squeeze(all_obs)
    elif len(np.shape(all_obs)) == 1:
        all_obs = np.expand_dims(all_obs, axis=1)
    if trajectories is None:
        trajectories = np.load(os.path.join(dir, "full_opt_trajectories.npy"))
    n_traj, l_traj, _ = np.shape(trajectories)
    _, d_states = np.shape(all_obs)
    paths = []
    for i in range(n_traj):
        current_trajectory = trajectories[i]
        current_obs = np.zeros((l_traj, d_states))
        current_actions = np.zeros((l_traj, 1))
        for tind in range(l_traj):
            s, a = current_trajectory[tind]
            current_obs[tind] = all_obs[s]
            current_actions[tind, 0] = a
        current_path = {'observations': np.array(current_obs), 'actions': np.array(current_actions)}
        paths.append(current_path)
    return paths


def algoexecute(algo, env, budget, trials, nInit=1, projections=None):
    """
    EXECUTE THE GIVEN IRL ALGORITHM ON THE SPECIFIED ENVIRONMENT
    """
    import os
    import numpy as np

    # Number of initial samples randomly selected
    boirlobj = IRLObject(algo, env,
                         projections=projections)  # Object that stores trajectories, result directories, data directories etc

    # Load the initialization points. They are selected from regions of high NLL making the training challenging.
    initXs = np.load(os.path.join(boirlobj.configurations.getTrajectoryDir(), "myinitpoints.npy"))

    if algo == constants.AIRL or algo == constants.GCL:
        # Code for AIRL and GCL taken from the official repository: https://github.com/justinjfu/inverse_rl
        import tensorflow as tf
        from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
        from sandbox.rocky.tf.envs.base import TfEnv
        from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
        from inverse_rl.envs.env_utils import CustomGymEnv
        from rllab.envs.gym_env import GymEnv
        from inverse_rl.algos.irl_trpo import IRLTRPO
        from inverse_rl.utils.log_utils import rllab_logdir
        import gym
        import numpy as np
        import os

        # Specify the architecture for the reward network.
        # For fair comparison, we used the same model for reward as with BOIRL.
        rewardarch = None
        irl_inits = None

        # Consider each environment and load appropriate reward architecture.
        # Only valid for gridworld3d, vborlange, rborlange and point mass maze
        if env == constants.GRIDWORLD3D:
            import gym_sigmoid
            if algo == constants.AIRL:
                from inverse_rl.models.architectures import sigmoid_airl_net
                rewardarch = sigmoid_airl_net
                # Necessary to load init points for rewards.
                irl_inits = np.load(os.path.join(boirlobj.configurations.getTrajectoryDir(), "irl_weights.npy"),
                                    allow_pickle=True).tolist()
                policy_inits = np.load(os.path.join(boirlobj.configurations.getTrajectoryDir(), "policy_weights.npy"),
                                       allow_pickle=True).tolist()
            else:
                from inverse_rl.models.architectures import sigmoid_gcl_net
                irl_inits = [0, 0, 0]
                rewardarch = sigmoid_gcl_net
        elif env == constants.VIRTBORLANGE or env == constants.REALBORLANGE:
            import gym_sweden
            if algo == constants.AIRL:
                from inverse_rl.models.architectures import airl_linear_net
                rewardarch = airl_linear_net
                # Necessary to load init points for rewards.
                irl_inits = np.load(os.path.join(boirlobj.configurations.getTrajectoryDir(), "irl_weights.npy"),
                                    allow_pickle=True).tolist()
            else:
                from inverse_rl.models.architectures import gcl_linear_net
                rewardarch = gcl_linear_net
                irl_inits = [np.zeros((3, 1))]
        elif env == constants.MAZE:
            import gym
            if algo == constants.AIRL:
                irl_inits = np.load(os.path.join(boirlobj.configurations.getTrajectoryDir(), "irl_weights.npy"),
                                    allow_pickle=True).tolist()
                from inverse_rl.models.architectures import maze_net_2dim
                rewardarch = maze_net_2dim
            else:
                irl_inits = [np.zeros((2))]
                from inverse_rl.models.architectures import maze_net_2dim_gcl
                rewardarch = maze_net_2dim_gcl
        else:
            raise (NotImplementedError(
                "AIRL and GCL requires OpenAI Gym version of the environment. Only available for Gridworld3d, Virtual Borlange, Real Borlange and Point Mass Maze"))

        # Import the necessary algorithm
        if algo == constants.AIRL:
            from inverse_rl.models.airl_state import AIRL
        else:
            from inverse_rl.models.imitation_learning import GAN_GCL
        import numpy as np
        import os

        # This code expects environments as OpenAi Gyms.
        # So make sure you follow the Readme.md to install the correct OpenAI Gym version of the environments.
        if env == constants.MAZE:
            mygymenv = TfEnv(GymEnv(constants.get_gym(env), record_video=False, record_log=False))
        else:
            mygymenv = TfEnv(CustomGymEnv(constants.get_gym(env), record_video=False, record_log=False))

        # Load expert data
        if env == constants.FETCH:
            all_experts = get_experts_fetch(boirlobj.env)
        elif env == constants.MAZE:
            all_experts = boirlobj.env.trajectories
        else:
            all_experts = get_experts(boirlobj.configurations.getTrajectoryDir())

        # Iterate over trials
        for tr in np.arange(trials):
            if env == constants.VIRTBORLANGE or env == constants.REALBORLANGE:
                irl_inits[0] = np.expand_dims(initXs[tr % initXs.shape[0]], axis=1)
            elif env == constants.MAZE:
                irl_inits[0] = initXs[tr % initXs.shape[0]]
            elif env == constants.GRIDWORLD3D:
                irl_inits[0] = initXs[tr % initXs.shape[0]][1] * np.ones(1)
                irl_inits[1] = initXs[tr % initXs.shape[0]][0] * np.ones(1)
                irl_inits[2] = initXs[tr % initXs.shape[0]][2] * np.ones(1)
                if algo == constants.AIRL:
                    for ggg in np.arange(3, 9):
                        irl_inits[ggg] = policy_inits[ggg]
            elif env == constants.MAZE:
                raise NotImplementedError()

            inds = np.random.permutation(np.arange(len(all_experts)))  # [0:50]
            maxlength = len(all_experts[0]['actions'])
            experts = [all_experts[ind] for ind in inds]
            if algo == constants.AIRL:
                irl_model = AIRL(env=mygymenv, expert_trajs=experts, state_only=True, fusion=False, max_itrs=budget,
                                 reward_arch=rewardarch, discount=boirlobj.configurations.getDiscounts())
            else:
                irl_model = GAN_GCL(env_spec=mygymenv.spec, expert_trajs=experts, state_only=True,
                                    discount=boirlobj.configurations.getDiscounts(),
                                    discrim_arch_args={"ff_arch": rewardarch})
            policy = GaussianMLPPolicy(name='policy', env_spec=mygymenv.spec, hidden_sizes=(32, 32))

            # Training algorithm
            myalgo = IRLTRPO(
                init_irl_params=irl_inits,
                env=mygymenv,
                policy=policy,
                irl_model=irl_model,
                n_itr=budget,
                batch_size=10,
                max_path_length=maxlength,
                discount=boirlobj.configurations.getDiscounts(),
                store_paths=True,
                irl_model_wt=1.0,
                entropy_weight=0.1,
                zero_environment_reward=True,
                baseline=LinearFeatureBaseline(env_spec=mygymenv.spec),
            )

            # Do the actual training and store the weights of the reward network
            mydirname = os.path.join('Data', algo, 'db_' + env + '_' + algo, str(tr))
            with rllab_logdir(algo=myalgo, dirname=mydirname):
                with tf.Session():
                    myalgo.train(os.path.join(boirlobj.configurations.getResultDir(), "weights%d.npy" % tr))
            tf.reset_default_graph()
    else:
        # Actual code BOIRL
        import numpy as np
        import os

        # Get and store the ground truth NLL value
        if env == constants.FETCH:
            gt_lik, _ = boirlobj.env.get_likelihood(None)
        elif env == constants.MAZE:
            boirlobj.env.trial = -1
            boirlobj.env.algo = algo
            gt_lik = boirlobj.env.get_likelihood()
        elif env == constants.VIRTBORLANGE or env == constants.REALBORLANGE or env == constants.GRIDWORLD2D or env == constants.GRIDWORLD3D:
            gt_lik, _ = boirlobj.env.get_likelihood(boirlobj.fullTrajectories)
        else:
            gt_lik = boirlobj.env.get_likelihood(boirlobj.configurations.fullTrajectories)
        np.save(os.path.join(boirlobj.configurations.getResultDir(), "gt_lik.npy"), gt_lik)

        # Start BO
        for trial in np.arange(trials):
            ### Only applicable for taurbf.
            ### Sample artificial trajectories using a uniform policy
            # print("Generating Artificial Trajectories")
            if algo == constants.RHORBF:
                if env == constants.FETCH:
                    subsetTrajectories, subsetStartPos, subsetArtTrajs = boirlobj.env.get_subs(
                        boirlobj.configurations.getNTrajs())
                elif env == constants.MAZE:
                    subsetTrajectories, subsetStartPos, subsetArtTrajs = boirlobj.env.get_subs(
                        boirlobj.configurations.getNTrajs(), boirlobj.configurations.getNArtTrajs())
                else:
                    subinds = np.random.permutation(boirlobj.fullTrajectories.shape[0])[
                              0:boirlobj.configurations.projections]
                    subsetTrajectories = boirlobj.fullTrajectories[subinds]
                    subsetStartPos = boirlobj.fullStartpos[subinds]
                    subsetArtTrajs = None
                    for at in range(boirlobj.configurations.getNArtTrajs()):
                        temp_trajectories = boirlobj.env.artificial_trajectories(subsetTrajectories, subsetStartPos)
                        if subsetArtTrajs is None:
                            subsetArtTrajs = np.expand_dims(temp_trajectories, axis=0)
                        else:
                            subsetArtTrajs = np.append(subsetArtTrajs, np.expand_dims(temp_trajectories, axis=0),
                                                       axis=0)
            else:
                subsetTrajectories = None
                subsetArtTrajs = None

            # Setup kernel
            boirlobj.setupkernel(subsetTrajectories, subsetArtTrajs)

            # Get the current initialization point
            initX = np.expand_dims(initXs[trial % initXs.shape[0]], axis=0)

            # Set a progress bar to observe the training progress
            boirlobj.pbar = tqdm(total=budget + nInit)
            boirlobj.pbar.set_description("Trial: %d" % trial)
            boirlobj.tr = trial

            # define the BO object
            myProblem = GPyOpt.methods.BayesianOptimization(boirlobj.blackboxfunc, boirlobj.bounds,
                                                            kernel=boirlobj.kernel,
                                                            model_type="GP", normalize_Y=False,
                                                            exact_feval=True, X=initX)  # initial_design_numdata=nInit)
            # Starting the optimization
            myProblem.run_optimization(budget)
            boirlobj.pbar.close()
            # Optimization over"
            # Save the results
            np.save(os.path.join(boirlobj.configurations.getResultDir(), "x_choices.npy"), np.array(boirlobj.x_choices))
            np.save(os.path.join(boirlobj.configurations.getResultDir(), "y_choices.npy"), np.array(boirlobj.y_choices))
            np.save(os.path.join(boirlobj.configurations.getResultDir(), "trial_tracks.npy"),
                    np.array(boirlobj.trial_track))
            np.save(os.path.join(boirlobj.configurations.getResultDir(), "length_tracks.npy"),
                    np.array(boirlobj.length_track))
            if len(boirlobj.stoch_pol) > 0:
                np.save(os.path.join(boirlobj.configurations.getResultDir(), "stochpols.npy"),
                        np.array(boirlobj.stoch_pol))
            np.save(os.path.join(boirlobj.configurations.getResultDir(), "kernel_length_" + str(trial) + ".npy"),
                    boirlobj.kernel.lengthscale.values)  # kernel length for debugging purposes
            np.save(os.path.join(boirlobj.configurations.getResultDir(), "best_x" + str(trial) + ".npy"),
                    myProblem.x_opt)  # best x value corresponding to the best reward encountered during training
            if env == constants.FETCH:
                np.save(os.path.join(boirlobj.configurations.getResultDir(), "train_fids.npy"),
                        np.array(boirlobj.train_fids))

            # plot the convergence plot. This also includes euclidean distance between subsequent points
            myProblem.plot_convergence(
                os.path.join(boirlobj.configurations.getResultDir(), "convergence_" + str(trial) + ".png"))

            boirlobj.plotAcquisition(bo=myProblem, trial=trial)

            # Before training, we specify a set of random values for the reward function parameters and store it as allW.
            # We can calculate the correlation between the ground truth reward function and allW
            # We avoid rborlange and vborlange due to the heavy memory requirement
            if not (env == constants.REALBORLANGE or env == constants.VIRTBORLANGE):
                kk = boirlobj.kernel.K(X=boirlobj.gtheta, X2=boirlobj.allW)
                np.save(os.path.join(boirlobj.configurations.getResultDir(), "kk_" + str(trial) + ".npy"), kk)
                # We can also store the mapping of allW to the latent space.
                if algo == constants.RHORBF:
                    np.save(os.path.join(boirlobj.configurations.getResultDir(), "proxies_" + str(trial) + ".npy"),
                            boirlobj.kernel._get_proxy(boirlobj.allW, None)[0])


def algosor(algo, env, projections=None):
    """
    MAIN FUNCTION TO CALCULATE THE ESOR AND NLL OF THE GIVEN ALGORITHM IN THE SPECIFIC ENVIRONMENT.
    """
    print("Calculating BOIRL ESOR and NLL")
    algo_sor = None
    algo_lik = None
    boirlobj = IRLObject(algo, env, projections)
    gt_trajs = boirlobj.fullTrajectories
    gt_spos = boirlobj.fullStartpos

    if algo == constants.AIRL or algo == constants.GCL:
        tr = 0
        while True:
            # load weights:
            weight_fname = os.path.join(boirlobj.configurations.getResultDir(), "weights%d.npy") % tr
            if not os.path.exists(weight_fname):
                break

            allweights = np.load(weight_fname).squeeze()
            # Gridworld env has some mismatch in the index between saved weight and the one used in the reward function.
            if env == constants.GRIDWORLD3D:
                tempshift = np.array(allweights)[:, 0]  # shift
                tempsteep = np.array(allweights)[:, 1]  # steep
                allweights[:, 0] = tempsteep
                allweights[:, 1] = tempshift
            nrew = allweights.shape[0]
            temp_sor = np.zeros(nrew)
            temp_lik = np.zeros(nrew)
            if env == constants.VIRTBORLANGE:
                randomstart = False
                n_testtrajs = 5000
                spos = gt_spos[0:5000]
            else:
                randomstart = False
                n_testtrajs = gt_trajs.shape[0]
                spos = None
            for wind in tqdm(np.arange(0, nrew), desc="Trial: %d" % tr):
                w = allweights[wind]
                # Set the current reward function in the environment
                boirlobj.env.set_reward(w)
                if env == constants.MAZE:
                    boirlobj.env.trial = tr
                    boirlobj.env.algo = algo
                # Generate trajectories using the optimal policy learned from current reward function
                traj, _, stochpolicy = boirlobj.env.generate_trajectories(n_trajectories=n_testtrajs,
                                                                          random_start=randomstart,
                                                                          startpos=spos)
                # Calculate the likelihood of the expert demonstrations under current trajectories
                if stochpolicy is not None:
                    lik = boirlobj.env.get_likelihood_from_policy(gt_trajs, stochpolicy)
                else:
                    lik = boirlobj.env.get_likelihood(savefile=False)

                # Reset the reward function of the environment to the expert's true reward function
                boirlobj.env.set_reward(boirlobj.gtheta.squeeze())

                temp_lik[wind] = lik
                if not (env == constants.REALBORLANGE):
                    # Calculate the expected sum of rewards
                    sor, _ = boirlobj.env.evaluate_expsor(traj)
                    temp_sor[wind] = sor

            if algo_sor is None:
                algo_sor = np.expand_dims(temp_sor, axis=0)
                algo_lik = np.expand_dims(temp_lik, axis=0)
            else:
                algo_sor = np.append(algo_sor, np.expand_dims(temp_sor, axis=0), axis=0)
                algo_lik = np.append(algo_lik, np.expand_dims(temp_lik, axis=0), axis=0)
            tr += 1
            np.save(os.path.join(boirlobj.configurations.getResultDir(), "sor.npy"), algo_sor)
            np.save(os.path.join(boirlobj.configurations.getResultDir(), "likelihood.npy"), algo_lik)
        # Save ESOR and NLL
        np.save(os.path.join(boirlobj.configurations.getResultDir(), "sor.npy"), algo_sor)
        np.save(os.path.join(boirlobj.configurations.getResultDir(), "likelihood.npy"), algo_lik)
    else:
        # Load all the data from BOIRL training.
        # Single set of file for all trials
        # trial_tracks.npy indicate which indices in x_choices.npy and y_choices.npy correspond to which trial
        trials = np.load(os.path.join(boirlobj.configurations.getResultDir(), "trial_tracks.npy"))
        X = np.load(os.path.join(boirlobj.configurations.getResultDir(), "x_choices.npy"))
        Y = np.load(os.path.join(boirlobj.configurations.getResultDir(), "y_choices.npy"))
        policies = None
        policy_location = os.path.join(boirlobj.configurations.getResultDir(), "stochpols.npy")
        if os.path.exists(policy_location):
            policies = np.load(policy_location)
        unique_trial, trcnts = np.unique(trials, return_counts=True)
        unique_trial = np.sort(unique_trial)
        trcnts = np.amax(trcnts)

        for tr in tqdm(unique_trial):
            # Calculate ESOR and NLL for each trial
            # Find the indices to extract for current trial
            trialInds = np.where(trials == tr)[0]
            """
            if algo_lik is not None:
                if not (len(trialInds) == np.shape(algo_lik)[1]):
                    break
            """
            validX = X[trialInds]
            validY = Y[trialInds]
            validP = None
            if policies is not None:
                validP = policies[trialInds]

            # NLL can be directly read off from validY as opposed to other algorithms
            bestY = np.minimum.accumulate(validY)
            bestY = np.append(bestY, bestY[-1].item() * np.ones(trcnts - len(bestY)))
            if algo_lik is None:
                algo_lik = np.expand_dims(bestY, axis=0)
            else:
                algo_lik = np.append(algo_lik, np.expand_dims(bestY, axis=0), axis=0)

            # Calculate ESOR
            # Not possible to calculate ESOR for Real Borlange as there is no ground truth reward
            if not (env == constants.REALBORLANGE or env == constants.FETCH):
                # You only need to evaluate when the best reward function thus far has changed
                validChangeLoc = bestY[1:] - bestY[0:-1]
                validChangeLoc = np.append(np.ones(1), validChangeLoc)
                temp_sor = np.zeros(len(bestY))
                for n in range(len(bestY)):
                    if not (validChangeLoc[n] == 0):
                        current_x = validX[n]

                        # Set the reward function to the learned reward function
                        boirlobj.env.set_reward(current_x.squeeze())
                        if env == constants.MAZE:
                            boirlobj.env.trial = tr
                            boirlobj.env.algo = algo

                        # get current policy
                        current_policy = None
                        if validP is not None:
                            current_policy = validP[n]

                        if env == constants.VIRTBORLANGE:
                            randomstart = False
                            n_testtrajs = 5000
                            if env == constants.FETCH:
                                spos = gt_trajs["observation"].shape[0]
                            else:
                                spos = gt_spos[0:5000]
                        else:
                            randomstart = False
                            if env == constants.FETCH:
                                n_testtrajs = gt_trajs["observation"].shape[0]
                            else:
                                n_testtrajs = gt_trajs.shape[0]
                            spos = None

                        # Generate trajectories using the learned reward function or learned policy if available
                        if current_policy is None:
                            traj, _, _ = boirlobj.env.generate_trajectories(n_trajectories=n_testtrajs,
                                                                            random_start=randomstart, startpos=spos)
                        else:
                            traj, _, _ = boirlobj.env.generate_trajectories_from_policy(n_trajectories=n_testtrajs,
                                                                                        random_start=randomstart,
                                                                                        startpos=spos,
                                                                                        stoch_policy=current_policy)

                        # traj, _, _ = boirlobj.env.generate_trajectories(n_trajectories=gt_spos.shape[0],
                        #                                                random_start=True)

                        # Set the reward back to the ground truth reward function
                        boirlobj.env.set_reward(boirlobj.gtheta.squeeze())

                        # Calculate the Expected Sum of Rewards (ESOR) for the new set of trajectories using ground truth reward
                        sor, _ = boirlobj.env.evaluate_expsor(traj)
                    temp_sor[n] = sor
                if algo_sor is None:
                    algo_sor = np.expand_dims(temp_sor, axis=0)
                else:
                    algo_sor = np.append(algo_sor, np.expand_dims(temp_sor, axis=0), axis=0)

    # Save ESOR and NLL
    np.save(os.path.join(boirlobj.configurations.getResultDir(), "sor.npy"), algo_sor)
    np.save(os.path.join(boirlobj.configurations.getResultDir(), "likelihood.npy"), algo_lik)


def algoplot(algos, env, percent, niter, plotme=True):
    """
    MAIN CODE TO PLOT ESOR and NLL for the given environment.
    The code also outputs the average number of iterations( and std) required for each algorithm to achieve the specified percentage of Expert's ESOR
    """
    linewd = 3
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)
    fontlabs = {'family': 'serif',
                'color': 'black',
                'weight': 'normal',
                'size': 30,
                'verticalalignment': 'center',
                'horizontalalignment': 'center'
                }
    salpha = 0.9
    catchup = []

    fig = plt.figure(10, figsize=(24., 13.5), dpi=80)

    # Ground truth esor and likelihood
    boirlobj = IRLObject(algos[0], env)

    gtsor_path = os.path.join(boirlobj.configurations.getResultDir(),"gt_sor.npy")
    if os.path.exists(gtsor_path):
        gtsor = np.load(gtsor_path)
    else:
        gtsor, _ = boirlobj.env.evaluate_expsor(boirlobj.fullTrajectories)
        np.save(os.path.join(boirlobj.configurations.getResultDir(), "gt_sor.npy"), gtsor)
    # input(gtsor)

    limit = percent * gtsor
    # For VBORLANAGE and MAZE, the esor is always negative.
    if env == constants.REALBORLANGE or env == constants.VIRTBORLANGE or env == constants.MAZE:
        limit = 2 * gtsor - limit  # Because [1 + (1-percent)]*gtsor

    # deb_sor = np.zeros((3,10,21))
    algoind = 0

    max_length = -1
    if not env == constants.REALBORLANGE:
        for algo in algos:
            config = Configurations(algo, env)
            # Plot SOR and calculate the iterations to reach percentage of expert's ESOR
            sor = np.load(os.path.join(config.getResultDir(), "sor.npy"))
            algoind += 1
            max_length = max(max_length, sor.shape[1])
            bval = []
            #Find at which iteration we cross the limit (% of ESOR)
            for b in sor:
                loc = np.where(b > limit)[0]
                if len(loc) > 0:
                    bval.append(loc[0])
            # input(bval)
            bmean = np.mean(bval)
            bstd = np.std(bval)
            blen = "%d out of %d" % (len(bval), len(sor))
            catchup.append([algo, blen, bmean, bstd])

            # plot only niter iterations
            sor = sor[:, 0: niter]
            sormax = exp_moving_average(sor, alpha=salpha)  # np.maximum.accumulate(algosor, axis=1)
            sor_mean = np.mean(sormax, axis=0)
            sor_std = np.std(sormax, axis=0)
            p1 = plt.fill_between(np.arange(len(sor_mean)), sor_mean - sor_std, sor_mean + sor_std, alpha=0.1)
            plt_color = np.array(p1.get_facecolor()[0])
            plt_color[-1] = 1.
            plt.plot(np.arange(len(sor_mean)), sor_mean, "-", label=constants.LEGENDS[algo], c=plt_color,
                     linewidth=linewd)
            if sor_mean.shape[0] < niter:
                diff = niter - sor_mean.shape[0]
                extra_x = np.arange(sor_mean.shape[0], niter)
                extra_y = sor_mean[-1] * np.ones((diff))
                plt.plot(extra_x, extra_y, "--", linewidth=linewd, c=plt_color)

        # save ESOR
        plt.plot(np.arange(niter), gtsor * np.ones(niter), "--", linewidth=linewd / 2, label="Ground Truth")
        plt.legend(fontsize=20)
        plt.xlabel("Number of iterations", fontdict=fontlabs, labelpad=30)
        plt.ylabel("ESOR", fontdict=fontlabs, labelpad=15)
        if plotme:
            plt.savefig("ESOR_%s.png" % env, bbox_inches="tight")
        plt.close("all")

    fig = plt.figure(11, figsize=(24., 13.5), dpi=80)
    # Repeat for NLL
    for algo in algos:
        config = Configurations(algo, env)
        lik = np.load(os.path.join(config.getResultDir(), "likelihood.npy"))[:, 0:niter]
        likmax = exp_moving_average(lik, alpha=salpha)  # np.maximum.accumulate(algosor, axis=1)
        lik_mean = np.mean(likmax, axis=0)
        lik_std = np.std(likmax, axis=0)
        p1 = plt.fill_between(np.arange(len(lik_mean)), lik_mean - lik_std, lik_mean + lik_std, alpha=0.1)
        plt_color = np.array(p1.get_facecolor()[0])
        plt_color[-1] = 1.
        plt.plot(np.arange(len(lik_mean)), lik_mean, "-", label=constants.LEGENDS[algo], c=plt_color, linewidth=linewd)
        if lik_mean.shape[0] < niter:
            diff = niter - lik_mean.shape[0]
            extra_x = np.arange(lik_mean.shape[0], niter)
            extra_y = lik_mean[-1] * np.ones((diff))
            plt.plot(extra_x, extra_y, "--", linewidth=linewd, c=plt_color)

    plt.legend(fontsize=20)
    plt.xlabel("Number of iterations", fontdict=fontlabs, labelpad=30)
    plt.ylabel("NLL", fontdict=fontlabs, labelpad=15)
    plt.xticks(np.arange(0,max(lik_mean.shape[0],niter),5))
    if plotme:
        plt.savefig("NLL_%s.png" % env, bbox_inches="tight")
    plt.close("all")


    # Output the number of iterations required to reach the specified percentage of Expert's ESOR
    print("Number of iterations required to reach %2.1f%% of Expert's ESOR:%2.1f" % (100 * percent, gtsor))
    print(tabulate(catchup,
                   headers=['Algo', 'Success Rate', "Mean no: of iters for successful cases", "Std of no: of iters"]))
    return catchup
