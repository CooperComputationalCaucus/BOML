import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from multiprocessing import Pool

from .util import UtilityFunction,ensure_rng
from .target_space import _hashable

from sklearn.cluster import KMeans
# TODO: write parallelization for several acquisition functions

class LocalOptimizer():
    ''' Class of helper functions for minimization (Class needs to be picklable)'''
    def __init__(self,ac,gp,y_max,bounds,method="L-BFGS-B"):
        self.ac = ac
        self.gp = gp
        self.y_max=y_max
        self.bounds=bounds
        self.method=method
    def func_max(self,x):
        return -self.ac(x.reshape(1, -1), gp=self.gp, y_max=self.y_max)
    def func_min(self,x):
        return self.ac(x.reshape(1, -1), gp=self.gp, y_max=self.y_max)
    def minimizer(self,x_try):
        return minimize(self.func_min,
                       x_try.reshape(1, -1),
                       bounds=self.bounds,
                       method="L-BFGS-B")
    def maximizer(self,x_try):
        res= minimize(self.func_max,
                       x_try.reshape(1, -1),
                       bounds=self.bounds,
                       method="L-BFGS-B")
        res.fun[0]=-1*res.fun[0]
        return res

class LocalConstrainedOptimizer():
    ''' Class of helper functions for minimization (Class needs to be picklable)'''
    def __init__(self,ac,gp,y_max,bounds,method="SLSQP",constraints=()):
        self.ac = ac
        self.gp = gp
        self.y_max=y_max
        self.bounds=bounds
        self.method=method
        self.constraints=constraints
    def func_max(self,x):
        return -self.ac(x.reshape(1, -1), gp=self.gp, y_max=self.y_max)
    def func_min(self,x):
        return self.ac(x.reshape(1, -1), gp=self.gp, y_max=self.y_max)
    def minimizer(self,x_try):
        return minimize(self.func_min,
                       x_try.reshape(1, -1),
                       bounds=self.bounds,
                       method="L-BFGS-B")
    def maximizer(self,x_try):
        res= minimize(self.func_max,
                       x_try.reshape(1, -1),
                       bounds=self.bounds,
                       method=self.method,
                       constraints=self.constraints)
        res.fun=[-1*res.fun]
        return res    

def disc_acq_max(ac, instance, n_acqs=1, n_warmup=100000, n_iter=250, multiprocessing=1):
    """
    A function to find the maximum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (250) random starting points.

    Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    instance: DiscreteBayesianOptimization object instance.
    n_acqs: Integer number of acquisitions to take from acquisition function ac.
    n_warmup: number of times to randomly sample the aquisition function
    n_iter: number of times to run scipy.minimize
    multiprocessing: number of cores for multiprocessing of scipy.minimize

    Returns
    -------
    List of the arg maxs of the acquisition function.
    """

    # Inialization from instance
    gp = instance._gp
    y_max = instance._space.target.max()
    bounds = instance._space.bounds
    steps = instance._space.steps
    random_state = instance._random_state
    
    # Class of helper functions for minimization (Class needs to be picklable)
    lo = LocalOptimizer(ac,gp,y_max,bounds)
    
    # Warm up with random points
    x_tries = np.floor((random_state.uniform(bounds[:, 0], bounds[:, 1], 
                                             size=(n_warmup, bounds.shape[0]))-bounds[:,0])/
                                             steps)*steps+bounds[:,0]
    ys = ac(x_tries, gp=gp, y_max=y_max)
    
    # Using a dictionary to update top n_acqs,and retains the threshold for the bottom
    x_tries=x_tries[ys.argsort()[::-1]]
    ys= ys[ys.argsort()[::-1]]
    acqs = {}
    for idx in range(x_tries.shape[0]):
        if _hashable(x_tries[idx,:]) in instance.space:
            continue
        else:
            acqs[_hashable(x_tries[idx,:])]=ys[idx]
        if len(acqs)>n_acqs: break
    acq_threshold = sorted(acqs.items(), key=lambda t: (t[1],t[0]))[0]
   
    # Explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))
    
    if multiprocessing>1:
        # Memory unconscious multiprocessing makes list of n_iter results...
        pool = Pool(multiprocessing)
        results = list(pool.imap_unordered(lo.maximizer,x_seeds))
        pool.close()
        pool.join()
        for res in results:
            if not acqs or res.fun[0] >= acq_threshold[1]:
                if _hashable(instance.space._bin(res.x)) in instance.space:
                    continue
                acqs[_hashable(instance.space._bin(res.x))] = res.fun[0]
                if len(acqs)>n_acqs:
                    del acqs[acq_threshold[0]]
                    acq_threshold = sorted(acqs.items(), key=lambda t: (t[1],t[0]))[0]
    else:
        for x_try in x_seeds:
            # Maximize the acquisition function
            res = lo.maximizer(x_try)
            # See if success
            if not res.success:
                continue
    
            # Attempt to store it if better than previous maximum.
            # If it is new point, delete and replace threshold value
            if not acqs or res.fun[0] >= acq_threshold[1]:
                if _hashable(instance.space._bin(res.x)) in instance.space:
                    continue
                acqs[_hashable(instance.space._bin(res.x))] = res.fun[0]
                if len(acqs)>n_acqs:
                    del acqs[acq_threshold[0]]
                    acq_threshold = sorted(acqs.items(), key=lambda t: (t[1],t[0]))[0]

    return [key for key in acqs.keys()]

def disc_acq_KMBBO(ac, instance, n_acqs=1, n_slice=200, n_warmup=100000, n_iter=250, multiprocessing=1):
    """
    A function to find the batch sampled acquisition function. Uses slice sampling of continuous space,
    followed by k-means.The k- centroids are then binned and checked for redundancy. 
    slice: J. Artif. Intell. Res. 1, 1-24 (2017)
    slice+k-maens: arXiv:1806.01159v2
    
    Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    instance: DiscreteBayesianOptimization object instance.
    n_acqs: Integer number of acquisitions to take from acquisition function ac (the k in k-means).
    n_slice: integer number of slice samples (the data fed to k-means)
    n_warmup: number of times to randomly sample the aquisition function for a_min
    n_iter: number of times to run scipy.minimize for a_min
    multiprocessing: number of cores for multiprocessing of scipy.minimize

    Returns
    -------
    List of the sampled means of the acquisition function.
    """
    assert n_slice>=n_acqs, "number of points in slice (n_slice) must be greater \
                             than number of centroids in k-means (n_acqs)"
    
    # Inialization from instance
    gp = instance._gp
    y_max = instance._space.target.max()
    bounds = instance._space.bounds
    steps = instance._space.steps
    random_state = instance._random_state                         
    slice = np.zeros((n_slice,bounds.shape[0]))
    
    # Class of helper functions for optimization (Class needs to be picklable)
    lo = LocalOptimizer(ac,gp,y_max,bounds)
    
    # First finding minimum of acquisition function
    # Warm up with random points
    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_warmup, bounds.shape[0]))
    ys = ac(x_tries, gp=gp, y_max=y_max)
    a_min = ys.min()
    # Explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))
    if multiprocessing>1:
        # Memory unconscious multiprocessing makes list of n_iter results...
        pool = Pool(multiprocessing)
        results = list(pool.imap_unordered(lo.minimizer,x_seeds))
        pool.close()
        pool.join()
        a_min = sorted(results, key=lambda x: x.fun[0])[0].fun[0]
    else:
        for x_try in x_seeds:
            res = lo.minimizer(x_try)
            if not res.success:
                continue
            if a_min is None or res.fun[0] <= a_min:
                a_min = res.fun[0]
    if a_min>0: a_min = 0 # The algorithm will fail the minimum found is greater than 0

    # Initial sample over space
    s = random_state.uniform(bounds[:, 0], bounds[:, 1],size=(1, bounds.shape[0]))
    # Slice aggregation
    for i in range(n_slice):
        u = random_state.uniform(a_min,ac(s, gp=gp, y_max=y_max))
        while True:
            s = random_state.uniform(bounds[:, 0], bounds[:, 1],size=(1, bounds.shape[0]))
            if ac(s, gp=gp, y_max=y_max) > u:
                slice[i] = s
                break
    # Find centroids

    kmeans = KMeans(n_clusters = n_acqs,
                    random_state = random_state,
                    n_jobs = multiprocessing).fit(slice)
    # Make hashable and assert length
    acqs = {}
    for i in range(n_acqs):
        acqs[_hashable(instance.space._bin(kmeans.cluster_centers_[i,:]))] = i
    assert len(acqs) == n_acqs, "k-means clustering is not distinct in discretized space!" 
    
    return [key for key in acqs.keys()]                         
    
def disc_constrained_acq_max(ac, instance, n_acqs=1, n_warmup=100000, n_iter=250):    
    """
    A function to find the maximum of the acquisition function subject to inequality constraints

    It uses a combination of random sampling (cheap) and the 'SLSQP'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running SLSQP from `n_iter` (250) random starting points.
    #TODO: parallelization. Issues present in pickling constraint functions 
    
    Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    instance: DiscreteBayesianOptimization object instance.
    n_acqs: Integer number of acquisitions to take from acquisition function ac.
    n_warmup: number of times to randomly sample the aquisition function
    n_iter: number of times to run scipy.minimize

    Returns
    -------
    List of the arg maxs of the acquisition function.
    """
    # Inialization from instance
    gp = instance._gp
    y_max = instance._space.target.max()
    bounds = instance._space.bounds
    steps = instance._space.steps
    random_state = instance._random_state
    
    # Class of helper functions for minimization (Class needs to be picklable)
    lo = LocalConstrainedOptimizer(ac,gp,y_max,bounds,constraints=instance.get_constraint_dict())

    # Warm up with random points
    x_tries = np.floor((random_state.uniform(bounds[:, 0], bounds[:, 1], 
                                             size=(n_warmup, bounds.shape[0]))-bounds[:,0])/
                                             steps)*steps+bounds[:,0]
    # Apply constraints
    for dict in instance.get_constraint_dict():
        mask =np.ones((x_tries.shape[0],),dtype=bool)
        for i,x in enumerate(x_tries[:]):
            if dict['fun'](x)<0: mask[i]=False
        x_tries=x_tries[mask,:]
    ys = ac(x_tries, gp=gp, y_max=y_max)
    
    # Using a dictionary to update top n_acqs,and retains the threshold for the bottom
    x_tries=x_tries[ys.argsort()[::-1]]
    ys= ys[ys.argsort()[::-1]]
    acqs = {}
    for idx in range(x_tries.shape[0]):
        if _hashable(x_tries[idx,:]) in instance.space:
            continue
        else:
            acqs[_hashable(x_tries[idx,:])]=ys[idx]
        if len(acqs)>n_acqs: break
    acq_threshold = sorted(acqs.items(), key=lambda t: (t[1],t[0]))[0]
   
    # Explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))
    # Ensure seeds satisfy initial constraints
    for dict in instance.get_constraint_dict():
        mask = np.ones((x_seeds.shape[0],),dtype=bool)
        for i,x in enumerate(x_seeds[:]):
            if dict['fun'](x)<0: mask[i]=False
        x_seeds=x_seeds[mask,:]
        
    for x_try in x_seeds:
        # Maximize the acquisition function
        res = lo.maximizer(x_try)
        # See if success
        if not res.success:
            continue

        # Attempt to store it if better than previous maximum.
        # If it is new point, delete and replace threshold value
        if not acqs or res.fun[0] >= acq_threshold[1]:
            if _hashable(instance.space._bin(res.x)) in instance.space:
                continue
            acqs[_hashable(instance.space._bin(res.x))] = res.fun[0]
            if len(acqs)>n_acqs:
                del acqs[acq_threshold[0]]
                acq_threshold = sorted(acqs.items(), key=lambda t: (t[1],t[0]))[0]
    return [key for key in acqs.keys()]
    
def disc_constrained_acq_KMBBO(ac, instance, n_acqs=1, n_slice=200, n_warmup=100000, n_iter=250, multiprocessing=1):
    """
    A function to find the batch sampled acquisition function. Uses slice sampling of continuous space,
    followed by k-means.The k- centroids are then binned and checked for redundancy. 
    slice: J. Artif. Intell. Res. 1, 1-24 (2017)
    slice+k-maens: arXiv:1806.01159v2
    
    Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    instance: DiscreteBayesianOptimization object instance.
    n_acqs: Integer number of acquisitions to take from acquisition function ac (the k in k-means).
    n_slice: integer number of slice samples (the data fed to k-means)
    n_warmup: number of times to randomly sample the aquisition function for a_min
    n_iter: number of times to run scipy.minimize for a_min
    multiprocessing: number of cores for multiprocessing of scipy.minimize

    Returns
    -------
    List of the sampled means of the acquisition function.
    """
    assert n_slice>=n_acqs, "number of points in slice (n_slice) must be greater \
                             than number of centroids in k-means (n_acqs)"
    
    # Inialization from instance
    gp = instance._gp
    y_max = instance._space.target.max()
    bounds = instance._space.bounds
    steps = instance._space.steps
    random_state = instance._random_state                         
    slice = np.zeros((n_slice,bounds.shape[0]))
    constraint_dict = instance.get_constraint_dict()
    # Uses LBGFS for minding min (could be outside of constraints)
    lo = LocalOptimizer(ac,gp,y_max,bounds)
    
    # First finding minimum of acquisition function
    # Warm up with random points
    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_warmup, bounds.shape[0]))
    ys = ac(x_tries, gp=gp, y_max=y_max)
    a_min = ys.min()
    # Explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))
    if multiprocessing>1:
        # Memory unconscious multiprocessing makes list of n_iter results...
        pool = Pool(multiprocessing)
        results = list(pool.imap_unordered(lo.minimizer,x_seeds))
        pool.close()
        pool.join()
        a_min = sorted(results, key=lambda x: x.fun[0])[0].fun[0]
    else:
        for x_try in x_seeds:
            res = lo.minimizer(x_try)
            if not res.success:
                continue
            if a_min is None or res.fun[0] <= a_min:
                a_min = res.fun[0]
    if a_min>0: a_min = 0 # The algorithm will fail the minimum found is greater than 0

    
    # Initial sample over space
    invalid=True
    while invalid:
        s = random_state.uniform(bounds[:, 0], bounds[:, 1],size=(1, bounds.shape[0]))
        invalid=False
        for dict in constraint_dict:
            if dict['fun'](s.squeeze())<0: invalid=True   
    # Slice aggregation
    for i in range(n_slice):
        u = random_state.uniform(a_min,ac(s, gp=gp, y_max=y_max))  
        while True:
            invalid=True
            while invalid:
                s = random_state.uniform(bounds[:, 0], bounds[:, 1],size=(1, bounds.shape[0]))
                invalid=False
                for dict in constraint_dict:
                    if dict['fun'](s.squeeze())<0: invalid=True 
            if ac(s, gp=gp, y_max=y_max) > u:
                slice[i] = s
                break
    # Find centroids
    kmeans = KMeans(n_clusters = n_acqs,
                    random_state = random_state,
                    n_jobs = multiprocessing).fit(slice)
    # Make hashable and assert length
    acqs = {}
    for i in range(n_acqs):
        acqs[_hashable(instance.space._bin(kmeans.cluster_centers_[i,:]))] = i
    assert len(acqs) == n_acqs, "k-means clustering is not distinct in discretized space!" 
    return [key for key in acqs.keys()]          
    
    