import GPy
import numpy as np
from GPy.kern.src.rbf import RBF
from paramz.caching import Cache_this
from scipy.special import logsumexp
import timeit

class TauRBF(RBF):
    def __init__(self,input_dim,features,demonstrations,art_demonstrations,discount,lengthscale=None):
        #print("Using custom kernel")
        self.input_dim = input_dim
        self.features = features
        self.demos = demonstrations
        self.art_demos = art_demonstrations
        self.discount = discount
        #if not(np.all(demonstrations == None)):
        #    assert(len(demonstrations) == len(art_demonstrations))
        super(TauRBF, self).__init__(input_dim,lengthscale=lengthscale)

    def _get_proxy(self,X,X2=None):
        #return X,X2

        n_art_trajectories, n_trajectories, l_trajectory, _ = np.shape(self.art_demos)
        discounts = [self.discount ** t for t in range(l_trajectory)]
        discounts_x = np.repeat(np.expand_dims(discounts, axis=0), len(X), 0)
        discounts_art_x = np.repeat(np.expand_dims(np.repeat(np.expand_dims(discounts, axis=0), n_art_trajectories, 0), axis=0), len(X), 0)
        #Xsquared = np.square(X)
        #Xsquared[:,1] = np.array(X[:,1])

        #stime = timeit.default_timer()
        Rx = 10./(1. + np.exp(-1*X[:,0:1]*(np.repeat(np.expand_dims(self.features,axis=0),len(X),axis=0) - X[:,1:2])))
        Rx[:, len(self.features) - 1] = 0
        Rx = Rx + X[:,2:3]
        """
        stime2 = timeit.default_timer()
        inside2 = np.zeros((len(X),len(self.features)))
        for xind, x in enumerate(X):
            inside2[xind] = 1./(1.+np.exp(-1.*x[0]*(self.features - x[1])))
        stime3 = timeit.default_timer()
        print("Time Ratio %f"%((stime3-stime2)/(stime2-stime)))
        print("sss")
        """


        if X2 is not None:
            discounts_x2 = np.repeat(np.expand_dims(discounts, axis=0), len(X2), 0)
            discounts_art_x2 = np.repeat(
                np.expand_dims(np.repeat(np.expand_dims(discounts, axis=0), n_art_trajectories, 0), axis=0), len(X2), 0)
            #X2squared = np.square(X2)
            #X2squared[:,1] = X2[:,1]
            Rx2 = 10./(1. + np.exp(-1*X2[:,0:1]*(np.repeat(np.expand_dims(self.features,axis=0),len(X2),axis=0) - X2[:,1:2])))
            Rx2[:, len(self.features) - 1] = 0
            Rx2 = Rx2 + X2[:,2:3]
        proxy_x = []
        proxy_x2 = []

        demo_states = self.demos[:,:,0].astype(int)
        art_demo_states = self.art_demos[:,:,:,0].astype(int)
        for t in range(n_trajectories):
            Rtau_x = Rx[:, demo_states[t, :]]
            Rarttau_x = Rx[:, art_demo_states[:,t, :]]
            discounted_Rtau_x = np.multiply(Rtau_x, discounts_x)
            discounted_Rtau_x = np.expand_dims(np.sum(discounted_Rtau_x,axis=1),axis=1)
            discounted_Rarttau_x = np.multiply(Rarttau_x, discounts_art_x)
            discounted_Rarttau_x = np.sum(discounted_Rarttau_x,axis=2)
            denominator = np.append(discounted_Rtau_x, discounted_Rarttau_x, axis=1)

            """
            stime = timeit.default_timer()
            current_proxy_x = softmax(denominator,axis=1)[:,0]
            print("Using Softmax: %f" %(timeit.default_timer()-stime))
            stime2 = timeit.default_timer()
            tosub = np.max(denominator,axis=1,keepdims=True)
            #tosub2 = np.repeat(tosub,repeats=n_art_trajectories+1,axis=1)
            den = logsumexp(denominator-tosub,axis=1)
            #den2 = logsumexp(denominator-tosub2,axis=1)
            fff = np.exp(np.squeeze(discounted_Rtau_x-tosub) - den)
            print("Using np divide: %f" % (timeit.default_timer() - stime2))
            """
            tosub = np.max(denominator, axis=1, keepdims=True)
            den = logsumexp(denominator - tosub, axis=1)
            current_proxy_x = np.squeeze(discounted_Rtau_x-tosub)-den
            ####UNCOMMENT THIS IF YOU WANNA USE THE EXP OF REWARDS#########
            #current_proxy_x = np.exp(current_proxy_x)

            #current_proxy_x = np.exp(
            #    np.sum(np.multiply(Rtau_x, discounts_x) - np.multiply(Rtau_x, discounts_x), axis=1))
            proxy_x.append(current_proxy_x)
            if X2 is not None:
                Rtau_x2 = Rx2[:, demo_states[t, :]]
                Rarttau_x2 = Rx2[:, art_demo_states[:,t, :]]
                discounted_Rtau_x2 = np.multiply(Rtau_x2, discounts_x2)
                discounted_Rtau_x2 = np.expand_dims(np.sum(discounted_Rtau_x2, axis=1), axis=1)
                discounted_Rarttau_x2 = np.multiply(Rarttau_x2, discounts_art_x2)
                discounted_Rarttau_x2 = np.sum(discounted_Rarttau_x2, axis=2)
                denominator2 = np.append(discounted_Rtau_x2, discounted_Rarttau_x2, axis=1)
                tosub2 = np.max(denominator2, axis=1, keepdims=True)
                den2 = logsumexp(denominator2 - tosub2, axis=1)
                current_proxy_x2 = np.squeeze(discounted_Rtau_x2 - tosub2) - den2
                #current_proxy_x2 = np.exp(current_proxy_x2)

                #current_proxy_x2 = np.exp(
                #    np.sum(np.multiply(Rtau_x2, discounts_x2) - np.multiply(Rsumtau_x2, discounts_x2), axis=1))
                proxy_x2.append(current_proxy_x2)
        proxy_x = np.array(proxy_x).T
        if X2 is not None:
            proxy_x2 = np.array(proxy_x2).T
        else:
            proxy_x2 = None
        return proxy_x,proxy_x2


    @Cache_this(limit=3, ignore_args=())
    def _scaled_dist(self, X, X2=None):
        """
        Efficiently compute the scaled distance, r.

        ..math::
            r = \sqrt( \sum_{q=1}^Q (x_q - x'q)^2/l_q^2 )

        Note that if thre is only one lengthscale, l comes outside the sum. In
        this case we compute the unscaled distance first (in a separate
        function for caching) and divide by lengthscale afterwards

        """
        proxy_x, proxy_x2 = self._get_proxy(X, X2)
        if self.ARD:
            if proxy_x2 is not None:
                proxy_x2 = proxy_x2 / self.lengthscale
            return self._unscaled_dist(proxy_x / self.lengthscale, proxy_x2)
        else:
            return self._unscaled_dist(proxy_x, proxy_x2) / self.lengthscale

    def deb(self,X,X2):
        proxy_x, proxy_x2 = self._get_proxy(X, X2)
        usc = self._unscaled_dist(proxy_x, proxy_x2)
        ssc = self._unscaled_dist(proxy_x, proxy_x2) / self.lengthscale
        ssc2 = ssc**2
        essc2 = np.exp(ssc2)
        v = self.variance
        k = self.K(X,X2)
        return proxy_x,proxy_x2,usc,ssc,ssc2,essc2,v,k

    def gradients_X(self,dL_dK, X, X2=None):
        super(TauRBF, self).gradients_X(dL_dK,X,X2)

    def Kdiag(self, X):
        proxy_x,_ = self._get_proxy(X,None)
        ret = np.empty(proxy_x.shape[0])
        ret[:] = self.variance
        return ret

    def update_gradients_full(self, dL_dK, X, X2=None, reset=True):
        """
        Given the derivative of the objective wrt the covariance matrix
        (dL_dK), compute the gradient wrt the parameters of this kernel,
        and store in the parameters object as e.g. self.variance.gradient
        """
        assert(not self.ARD)
        self.variance.gradient = np.sum(self.K(X, X2)* dL_dK)/self.variance

        #now the lengthscale gradient(s)
        dL_dr = self.dK_dr_via_X(X, X2) * dL_dK

        if self.ARD:
            tmp = dL_dr*self._inv_dist(X, X2)
            if X2 is None: X2 = X
            if use_stationary_cython:
                self.lengthscale.gradient = self._lengthscale_grads_cython(tmp, X, X2)
            else:
                self.lengthscale.gradient = self._lengthscale_grads_pure(tmp, X, X2)
        else:
            r = self._scaled_dist(X, X2)
            self.lengthscale.gradient = -np.sum(dL_dr*r)/self.lengthscale

        if self.use_invLengthscale: self.inv_l.gradient = self.lengthscale.gradient * (self.lengthscale ** 3 / -2.)

"""
G = Gridworld(5,0,0.9)
trajectories, start_states = G.generate_trajectory(random_start=True)
artificial_trajectories = G.artificial_trajectories(trajectories,start_states)


X = np.random.uniform(-3.,3.,(20,1))
Y = np.sin(X) + np.random.randn(20,1)*0.05

#kernel = TauRBF(3,G.features,trajectories,artificial_trajectories,G.discount)
kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
print("Before:")
print(kernel.lengthscale.values)
m = GPy.models.GPRegression(X,Y,kernel)
m.optimize(messages=True)
print("After:")
print(kernel.lengthscale.values)
print("ssss")
"""

"""
X = np.random.uniform(-3.,3.,(20,3))
Y = np.sin(X) + np.random.randn(20,1)*0.05
kernel = TauRBF(input_dim=3,features=G.features,demonstrations=trajectories,art_demonstrations=artificial_trajectories,discount=G.discount)
#kernel = GPy.kern.RBF(input_dim=3, variance=1., lengthscale=1.)
m = GPy.models.GPRegression(X,Y,kernel)
m.optimize(messages=True)
"""
