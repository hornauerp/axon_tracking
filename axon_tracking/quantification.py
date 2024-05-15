import sklearn, sys
import numpy as np
from scipy.spatial.distance import pdist, cdist
import scipy.stats as stats
import warnings
sys.path.append("/home/phornauer/Git/axon_tracking/")
import axon_tracking.skeletonization as skel


def get_sliding_window_velocity(path_list,params,window_size=5,min_r2=0.95):
    warnings.simplefilter("ignore")
    all_vels = []
    num_vals = [] #count number of values for each velocity calc
    target_point_list = []
    for path in path_list:
        vels, r2s = [], []
        for i in range(path.shape[0]-window_size+1):
            target_points = path[i+1:i+window_size,:]
            if not any(np.array_equal(target_points,tp) for tp in target_point_list):
                dists = cdist(path[i:i+1,:2],target_points[:,:2]) / 1000
                time = (target_points[:,2] - path[i,2]) / (params['sampling_rate']/1000)
                regressor = sklearn.linear_model.LinearRegression(fit_intercept=False)
                vel_x = time.reshape(-1,1)
                vel_y = dists.reshape(-1,1)
                regressor.fit(vel_x, vel_y)
                y_pred = regressor.predict(vel_x)
                r2 = sklearn.metrics.r2_score(vel_y, y_pred)
                if not np.isnan(regressor.coef_[0][0]):
                    vels.append(regressor.coef_[0][0])
                    r2s.append(r2)
                target_point_list.append(target_points)
                
        filtered_vel = np.mean(np.abs([vels[x] for x in range(len(vels)) if r2s[x] > min_r2]))
        if not np.isnan(filtered_vel):
            all_vels.append(filtered_vel)
            num_vals.append(len(vels))
        
    mean_vel = np.average(all_vels,weights=num_vals)
    variance = np.average((all_vels - mean_vel)**2, weights=num_vals)
    std_vel = np.sqrt(variance)
    return mean_vel, std_vel

def get_simple_template_size(branches):
    template_size = np.unique(np.concatenate(branches)).shape[0]
    return template_size

def get_branch_point_count(skeleton):
    branch_points = len(skeleton.branches())
    return branch_points

def get_branch_point_dists(skeleton):
    ais = get_ais(skeleton)
    branch_points = skeleton.vertices[skeleton.branches(),:2]
    dists = cdist(ais,branch_points) / 1000
    return dists

def get_branch_lengths(branches):
    lengths = []
    for branch in branches:
        branch_diff = np.diff(branch,axis=0)
        dist = np.sqrt(branch_diff[:,0]**2 + branch_diff[:,1]**2) / 1000
        lengths.append(np.sum(dist))
    return lengths

def get_longest_path(skeleton):
    longest_path = np.max([len(x) for x in skeleton.paths()])
    return longest_path

def get_projection_dists(skeleton):
    ais = get_ais(skeleton)
    terminals = skeleton.vertices[skeleton.terminals(),:2]
    dists = cdist(ais,terminals) / 1000
    return dists

def get_ais(skeleton):
    ais_idx = np.argmin(skeleton.vertices[:,2])
    ais = skeleton.vertices[ais_idx:ais_idx+1,:2]
    return ais

def get_terminal_count(skeleton):
    terminal_count = len(skeleton.terminals())
    return terminal_count