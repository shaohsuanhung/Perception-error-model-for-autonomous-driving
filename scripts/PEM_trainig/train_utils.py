import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyhhmm.gaussian import GaussianHMM
from pyhhmm.multinomial import MultinomialHMM
from pyhhmm.heterogeneous import HeterogeneousHMM
from prettytable import PrettyTable
import pyhhmm.utils as hu
import math
from sklearn.model_selection import train_test_split
# from HMM_utils import  CalculateInitalProb_row,CalculateTransitionMtx_row,Calultae_Mean_Variance_row

# Calculate the Initial state, each object in the first element
def CalculateInitalProb(df):
    vis1 = 0 
    vis2 = 0
    vis3 = 0
    vis4 = 0

    for idx in range(1,df.shape[1],4):
        # Only count the first element in that column
        vis = df.iloc[0,idx+3]
        # Write switch case from vls 1 to 4
        if vis == 1:
            vis1 += 1
        elif vis == 2:
            vis2 += 1
        elif vis == 3:
            vis3 += 1
        elif vis == 4:
            vis4 += 1
        else:
            print("Error")
    print('VIS1: ',vis1,'VIS2: ',vis2,'VIS3: ',vis3,'VIS4: ',vis4)
    initla_state = np.array([vis1/((df.shape[1]-1)/4),vis2/((df.shape[1]-1)/4)\
                            ,vis3/((df.shape[1]-1)/4),vis4/((df.shape[1]-1)/4)])

    return initla_state 

# Calculate the Transition Matrix
def CalculateTransitionMtx(df,visualize = False,n_states = None):
    # If n_states is not specific, then the default is 4
    if n_states == None or n_states == 4:
        n_states = 4
        states = ['VIS1','VIS2','VIS3','VIS4']
        Transition_Mtx = [[0]*4 for _ in range(4)]
    else:
        states = [str(i) for i in range(1,n_states)]
        Transition_Mtx = [[0]*states for _ in range(states)]

    # Count the number of state change the Transition Matrix from the given df (supervised learning)
    for idx in range(1,df.shape[1],4):
        visibility_sequence = df.iloc[:,idx+3]
        for (i,j) in zip(visibility_sequence,visibility_sequence[1:]):
            # To avoid the NaN value (last value when the number of seq. is odd)
            if str(j) != 'nan':
                Transition_Mtx[int(i)-1][int(j)-1] += 1
            else:
                continue

    # Convert the count to probability
    for row in Transition_Mtx:
        row[:] = [x / sum(row) for x in row]

    # Visualize the Transition Matrix if visualize is True
    if visualize:
        rows = []
        for i, row in enumerate(Transition_Mtx):
            rows.append([states[i]] + ['P({}|{})={:.3f}'.format(states[j], states[i], tp)
                        for j, tp in enumerate(row)])
        t = PrettyTable(['_']+states)
        for row in rows:
            t.add_row(row)
        print(t)

    return Transition_Mtx

# Calculate the means Emission Matrix
# 1. Read visibility each object, 
# 2. add up the r and theta value in corresponding mean of state
def Calultae_Mean_Variance(df,visaualize = False):
    noise_mean = [[0]*2 for _ in range(4)]
    nbr_of_state = [[0] for _ in range(4)]
    r_list = [list() for _ in range(4)]
    theta_list = [list() for _ in range(4)]
    for idx in range(1,df.shape[1],4):
        noise_r_obj = df.iloc[:,idx]
        noise_theta_obj = df.iloc[:,idx+1]
        vis = df.iloc[:,idx+3]
        for (r,theta,vis) in zip(noise_r_obj,noise_theta_obj,vis):
            if str(vis) != 'nan':
                nbr_of_state[int(vis)-1][0] += 1
                noise_mean[int(vis)-1][0] += r
                noise_mean[int(vis)-1][1] += theta
                r_list[int(vis)-1].append(r)
                theta_list[int(vis)-1].append(theta)
            else:
                continue
    # Calculate the Full covariance
    noise_cov  = list()
    for i in range(4):
        noise_cov.append(np.cov(r_list[i],theta_list[i]))
    # Calculate the mean
    noise_mean = [[noise_mean[i][j]/nbr_of_state[i][0] for j in range(2)] for i in range(4)]
    if visaualize:
        print('Means')
        states = ['VIS1','VIS2','VIS3','VIS4']
        rows = []
        emissions = ['r', 'theta']
        for i, row in enumerate(noise_mean):
            rows.append([states[i]] + ['{:.3f}'.format(ep) for ep in row])
        t = PrettyTable(['_'] + ['r', 'theta'])
        for row in rows:
            t.add_row(row)
        print(t)
        print('Covariances')
        for ns, state in enumerate(states):
            print(state)
            rows = []
            for i, row in enumerate(noise_cov[ns]):
                rows.append([emissions[i]] + ['{:.3f}'.format(ep) for ep in row])
            
            t = PrettyTable(['_'] + ['r', 'theta'])
            for row in rows:
                t.add_row(row)
            print(t)
            
    return noise_mean, noise_cov

################ From Row based dataframe ################
def CalculateInitalProb_row(df):
    vis1 = 0 
    vis2 = 0
    vis3 = 0
    vis4 = 0

    for idx in range(0,df.shape[0]):
        # Only the count the first element
        vis_seq = list(df.iloc[idx,3].replace("[","").replace("]","").split(","))
        vis_seq = [int(i) for i in vis_seq]
        vis = vis_seq[0]
        # Write switch case from vis 1 to 4
        if vis == 1:
            vis1 += 1
        elif vis == 2:
            vis2 += 1
        elif vis == 3:
            vis3 += 1
        elif vis == 4:
            vis4 += 1
        else:
            print("Error")

    # initla_state = np.array([vis1/df.shape[0],vis2/df.shape[0]\
    #                         ,vis3/df.shape[0],vis4/df.shape[0]])
    print('VIS1: ',vis1,'VIS2: ',vis2,'VIS3: ',vis3,'VIS4: ',vis4)
    initla_state = np.array([vis1/266,vis2/266\
                            ,vis3/266,vis4/266])
    return initla_state 

# Calculate the Transition Matrix
def CalculateTransitionMtx_row(df,visualize = False,n_states = None):
    # If n_states is not specific, then the default is 4
    if n_states == None or n_states == 4:
        n_states = 4
        states = ['VIS1','VIS2','VIS3','VIS4']
        Transition_Mtx = [[0]*4 for _ in range(4)]
    else:
        states = [str(i) for i in range(1,n_states)]
        Transition_Mtx = [[0]*states for _ in range(states)]

    # Count the number of state change the Transition Matrix from the given df (supervised learning)
    for idx in range(0,df.shape[0]):
        # visibility_sequence = df.iloc[idx,3]
        visibility_sequence = list(df.iloc[idx,3].replace("[","").replace("]","").split(","))
        visibility_sequence = [int(i) for i in visibility_sequence]
        for (i,j) in zip(visibility_sequence,visibility_sequence[1:]):
            # To avoid the NaN value (last value when the number of seq. is odd)
            if str(j) != 'nan':
                Transition_Mtx[int(i)-1][int(j)-1] += 1
            else:
                continue

    # Convert the count to probability
    for row in Transition_Mtx:
        row[:] = [x / sum(row) for x in row]

    # Visualize the Transition Matrix if visualize is True
    if visualize:
        rows = []
        for i, row in enumerate(Transition_Mtx):
            rows.append([states[i]] + ['P({}|{})={:.3f}'.format(states[j], states[i], tp)
                        for j, tp in enumerate(row)])
        t = PrettyTable(['_']+states)
        for row in rows:
            t.add_row(row)
        print(t)

    return Transition_Mtx

# Calculate the means Emission Matrix
# 1. Read visibility each object, 
# 2. add up the r and theta value in corresponding mean of state
def Calultae_Mean_Variance_row(df,visaualize = False):
    noise_mean = [[0]*2 for _ in range(4)]
    nbr_of_state = [[0] for _ in range(4)]
    r_list = [list() for _ in range(4)]
    theta_list = [list() for _ in range(4)]
    for idx in range(0,df.shape[0]):
        # noise_r_obj = df.iloc[idx,0]
        # noise_theta_obj = df.iloc[idx,1]
        # vis = df.iloc[idx,3]
        noise_r_obj = list(df.iloc[idx,0].replace("[","").replace("]","").split(","))
        noise_r_obj = [float(i) for i in noise_r_obj]
        noise_theta_obj = list(df.iloc[idx,1].replace("[","").replace("]","").split(","))
        noise_theta_obj = [float(i) for i in noise_theta_obj]
        vis = list(df.iloc[idx,3].replace("[","").replace("]","").split(","))
        vis = [int(i) for i in vis]
        for (r,theta,vis) in zip(noise_r_obj,noise_theta_obj,vis):
            if str(vis) != 'nan':
                nbr_of_state[int(vis)-1][0] += 1
                noise_mean[int(vis)-1][0] += r
                noise_mean[int(vis)-1][1] += theta
                r_list[int(vis)-1].append(r)
                theta_list[int(vis)-1].append(theta)
            else:
                continue
    # Calculate the Full covariance
    noise_cov  = list()
    for i in range(4):
        noise_cov.append(np.cov(r_list[i],theta_list[i]))
    # Calculate the mean
    noise_mean = [[noise_mean[i][j]/nbr_of_state[i][0] for j in range(2)] for i in range(4)]
    if visaualize:
        states = ['VIS1','VIS2','VIS3','VIS4']
        print('Means')
        rows = []
        emissions = ['r', 'theta']
        for i, row in enumerate(noise_mean):
            rows.append([states[i]] + ['{:.3f}'.format(ep) for ep in row])
        t = PrettyTable(['_'] + ['r', 'theta'])
        for row in rows:
            t.add_row(row)
        print(t)
        print('Covariances')
        for ns, state in enumerate(states):
            print(state)
            rows = []
            for i, row in enumerate(noise_cov[ns]):
                rows.append([emissions[i]] + ['{:.3f}'.format(ep) for ep in row])
            
            t = PrettyTable(['_'] + ['r', 'theta'])
            for row in rows:
                t.add_row(row)
            print(t)
            
    return noise_mean, noise_cov


def data_loader(datapath):
    df_train = pd.read_csv(datapath)
    # --------------------------------------------------------------#
    # Prepare noise r and theta data for unsupervised learning HMM1 # 
    # --------------------------------------------------------------#
    noise_observation = list()
    for idx in range(0,df_train.shape[0]):
        r_seq = list(df_train.iloc[idx,0].replace("[","").replace("]","").split(","))
        r_seq = [float(i) for i in r_seq]
        theta_seq = list(df_train.iloc[idx,1].replace("[","").replace("]","").split(","))
        theta_seq = [float(i) for i in theta_seq]
        noise_observation.append(np.array([r_seq,theta_seq]).T)

    # ----------------------------------------------------#
    # List of length of each sequence for model selection # 
    # ----------------------------------------------------#
    length = [len(seq) for seq in noise_observation]
    # print(length)
    # --------------------------------------------------------------#
    #  Prepare miss/detect data for unsupervised learning for HMM2  # 
    # --------------------------------------------------------------#
    MD_observation = list()
    for idx in range(0,df_train.shape[0]):
        seq = list(df_train.iloc[idx,2].replace("[","").replace("]","").split(","))
        seq = [int(i) for i in seq]
        MD_observation.append(np.array(seq).reshape((-1,1)))

    return noise_observation,length,MD_observation

def train_supervised_HMM1(df,visualize = False):
    states = ['VIS1', 'VIS2','VIS3','VIS4']
    emissions = ['r', 'theta']
    # instantiate a MultinomialHMM object
    Supervised_HMM1 = GaussianHMM(
        n_states=4,
        n_emissions=2, # noise of r and theta 
        covariance_type='full'
    )
    # the initial state probabilities, array of shape (n_states, )
    Supervised_HMM1.pi = CalculateInitalProb_row(df)
    # the state transition probabilities, array of shape (n_states, n_states)
    Supervised_HMM1.A = CalculateTransitionMtx_row(df, visualize = False,n_states = 4)
    Supervised_HMM1.means, Supervised_HMM1.covars = Calultae_Mean_Variance_row(df,visaualize = False)

    if(visualize):
        hu.pretty_print_hmm(Supervised_HMM1,states=states, hmm_type='Gaussian', emissions=emissions)
            
    return Supervised_HMM1

def train_unsupervised_HMM1(input_obs,model_selection = False):

    if model_selection:
        n_selection_states = [1,2,3,4,5,6,7,8,9,10,11]
        lengths = [len(seq) for seq in input_obs]
        n_samples = sum(lengths)
        models = []
        BestModel_UHMM1 = None
        # criteria = {'AIC': np.zeros_like(n_selection_states), 'BIC': np.zeros_like(n_selection_states)}
        criteria = {'AIC': np.zeros_like(n_selection_states)}
        for idx, ns in enumerate(n_selection_states):
            # Initialize a temp model
            temp_model = GaussianHMM(n_states=ns, n_emissions=2, 
                                    covariance_type='full', verbose=False)
            # Train the model with the current number of states
            temp_model, ll = temp_model.train(input_obs, 
                                            n_init = 1,
                                            n_iter=50, 
                                            conv_thresh=1e-3, 
                                            conv_iter=5, 
                                            no_init=False,
                                            plot_log_likelihood=False,
                                            ignore_conv_crit=False)
            # Compute the number of free parameters of the model
            dof = hu.get_n_fit_scalars(temp_model)
            
            # Compute the number of order selection criteria
            aic = hu.aic_hmm(ll,dof)
            criteria['AIC'][idx] = aic

            # bic = hu.bic_hmm(ll,n_samples,dof)
            # criteria['BIC'][idx] = bic

            # print('{} states: logL = {:.3f}, AIC = {:.3f}, BIC = {:.3f}'.format(ns, ll, aic, bic))
            print('{} states: logL = {:.3f}, AIC = {:.3f}'.format(ns, ll, aic))
            # Save the best model
            if idx == 0:
                print('Initial best model')
                best_aic = aic
                # best_bic = bic
                best_ll = ll
                BestModel_UHMM1 = temp_model
            else:
                if best_ll < ll:
                    print('Best model updated. Selecting model with {} states'.format(ns))
                    BestModel_UHMM1 = temp_model
                    best_ll = ll

        hu.plot_model_selection(n_selection_states, criteria)
        return BestModel_UHMM1, best_ll, criteria
    else:
        Unsupervised_HMM1 = GaussianHMM(
            n_states=9,
            n_emissions=2, # noise of r and theta 
            covariance_type='full',
            verbose=True
        )

        Unsupervised_HMM1, log_likelihoods = Unsupervised_HMM1.train(
            input_obs,
            n_init=1,     # number of initialisations to perform
            n_iter=150,   # maximum number of iterations to run
            conv_thresh=0.001,  # what percentage of change in the log-likelihood between iterations is considered convergence
            conv_iter=5,  # for how many iterations does it have to hold
            # whether to plot the evolution of the log-likelihood over the iterations
            plot_log_likelihood=True,
            # set to True if want to train until maximum number of iterations is reached
            ignore_conv_crit=False,
            no_init=False,  # set to True if the model parameters shouldn't be re-initialised befor training; in this case they have to be set manually first, otherwise errors occur
        )
        return Unsupervised_HMM1, log_likelihoods, None

def train_unsupervised_HMM2(prescence_obs,model_selection = False):

    if model_selection:
        n_selection_states = [1,2,3,4,5,6,7,8,9,10,11]
        # n_selection_states = [1,2,3,4,5,6,7]
        lengths = [len(seq) for seq in prescence_obs]
        n_samples = sum(lengths)
        models = []
        BestModel_UHMM2 = None
        # criteria = {'AIC': np.zeros_like(n_selection_states), 'BIC': np.zeros_like(n_selection_states)}
        criteria = {'AIC': np.zeros_like(n_selection_states)}

        for idx, ns in enumerate(n_selection_states):
            # Initialize a temp model
            temp_model = MultinomialHMM(n_states=ns, 
                                        n_emissions=1,
                                        n_features=[2],
                                        init_type='random',
                                        verbose=True)
            # Train the model with the current number of states
            temp_model, ll = temp_model.train(prescence_obs, 
                                            n_init = 1,
                                            n_iter=30, 
                                            conv_thresh=1e-3, 
                                            conv_iter=5, 
                                            no_init=False,
                                            plot_log_likelihood=False,
                                            ignore_conv_crit=False)
            # Compute the number of free parameters of the model
            dof = hu.get_n_fit_scalars(temp_model)
            
            # Compute the number of order selection criteria
            aic = hu.aic_hmm(ll,dof)
            criteria['AIC'][idx] = aic

            # bic = hu.bic_hmm(ll,n_samples,dof)
            # criteria['BIC'][idx] = bic

            print('{} states: logL = {:.3f}, AIC = {:.3f}'.format(ns, ll, aic))
            # print('{} states: logL = {:.3f}, AIC = {:.3f}'.format(ns, ll, aic))
            # Save the best model
            if idx == 0:
                best_aic = aic
                # best_bic = bic
                best_ll = ll
                BestModel_UHMM2 = temp_model
            else:
                if best_ll < ll:
                    print("Best Model updated. Select model with {} states".format(ns))
                    BestModel_UHMM2 = temp_model
                    best_ll = ll
        # plot the model order selection criteria in function of the number of states
        hu.plot_model_selection(n_selection_states, criteria)

        return BestModel_UHMM2, best_ll, criteria
    else:
        HMM2 = MultinomialHMM(n_states=2, 
                              n_emissions=1,
                              n_features=[2],
                              init_type='random',
                              verbose=True)
        HMM2, log_likelihoods = HMM2.train(prescence_obs,
                                           n_init = 1,
                                           n_iter=150, 
                                           conv_thresh=1e-4,
                                           conv_iter=10,
                                           plot_log_likelihood=True,
                                           no_init=False,)
        return HMM2, log_likelihoods, None

def PEMs_Evaluation(df_test, noise_HMM, MD_HMM):
    '''Evaluate the unsupervised HMM for noise and prescene given the test set'''
    
    inference_MD_seq = list()
    inference_noise_seq = list()
    gt_prescene_seq = list()
    gt_noise_seq = list()
    for idx in range(0,df_test.shape[0]):
        r_seq = list(df_test.iloc[idx,0].replace("[","").replace("]","").split(","))
        r_seq = [float(i) for i in r_seq]
        theta_seq = list(df_test.iloc[idx,1].replace("[","").replace("]","").split(","))
        theta_seq = [float(i) for i in theta_seq]
        nbr_sample = len(r_seq)
        # Sampling
        MDobs_seq = MD_HMM.sample(n_sequences=1,n_samples = nbr_sample,return_states=False)
        NoiseObs_seq = noise_HMM.sample(n_sequences=1,n_samples = nbr_sample,return_states=False)
        
        # Get GT from test set
        gt_MD_seq = list(df_test.iloc[idx,2].replace("[","").replace("]","").split(","))
        gt_MD_seq = [int(i) for i in gt_MD_seq]
        gt_NoiseObs_seq = list(df_test.iloc[idx,1].replace("[","").replace("]","").split(","))
        gt_NoiseObs_seq = [float(i) for i in gt_NoiseObs_seq]

        #  Clean up data to list sum(n_i) x 1
        inference_MD_seq.append(np.concatenate(MDobs_seq).ravel())
        inference_noise_seq.append(np.concatenate(NoiseObs_seq).ravel())
        gt_prescene_seq.append(gt_MD_seq)
        gt_noise_seq.append(gt_NoiseObs_seq)

    inference_MD_seq = np.concatenate(inference_MD_seq).ravel()
    inference_noise_seq = np.concatenate(inference_noise_seq).ravel()
    gt_prescene_seq = np.concatenate(gt_prescene_seq).ravel()
    gt_noise_seq = np.concatenate(gt_noise_seq).ravel()

    # return the return the inference and ground truth {MD,noise} sequence
    return inference_MD_seq,inference_noise_seq,gt_prescene_seq,gt_noise_seq

def Supervised_PEMs_Evaluation(df_test, noise_SHMM, MD_HMM):
    '''Evaluate the Supervised HMM for noise and unsupervised prescene HMM given the test set'''
    
    inference_MD_seq = list()
    inference_noise_seq = list()
    gt_prescene_seq = list()
    gt_noise_seq = list()
    for idx in range(0,df_test.shape[0]): # Do the inference row by row (obj traj. by obj traj.)
        r_seq = list(df_test.iloc[idx,0].replace("[","").replace("]","").split(","))
        r_seq = [float(i) for i in r_seq]
        nbr_sample = len(r_seq)
        # Sampling
        MDobs_seq = MD_HMM.sample(n_sequences=1,n_samples = nbr_sample,return_states=False)
        # Sampling from the supervised model
        NoiseObs_seq, NoiseObs_states = noise_SHMM.sample(n_sequences=1,n_samples = nbr_sample,return_states=True)
        
        # Get GT from test set
        gt_MD_seq = list(df_test.iloc[idx,2].replace("[","").replace("]","").split(","))
        gt_MD_seq = [int(i) for i in gt_seq]
        gt_NoiseObs_seq = list(df_test.iloc[idx,1].replace("[","").replace("]","").split(","))
        gt_NoiseObs_seq = [float(i) for i in gt_seq]

        #  Clean up data to list sum(n_i) x 1
        inference_MD_seq.append(np.concatenate(obs_seq).ravel())
        inference_noise_seq.append(np.concatenate(NoiseObs_seq).ravel())
        gt_prescene_seq.append(gt_MD_seq)
        gt_noise_seq.append(gt_NoiseObs_seq)

    inference_MD_seq = np.concatenate(inference_MD_seq).ravel()
    inference_noise_seq = np.concatenate(inference_noise_seq).ravel()
    gt_prescene_seq = np.concatenate(gt_prescene_seq).ravel()
    gt_noise_seq = np.concatenate(gt_noise_seq).ravel()

    # return the return the inference and ground truth {MD,noise} sequence
    return inference_MD_seq,inference_noise_seq,gt_prescene_seq,gt_noise_seq