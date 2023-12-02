import numpy as np
import pandas as pd
from prettytable import PrettyTable

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
