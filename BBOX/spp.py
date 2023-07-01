import numpy as np

# Xuhui Fan
# 11-Dec-2022

def spp(tau_val = 1 / 64, length_val = np.array([256, 256]), theta_val = np.array([0.93, 0.93])):
    # Generate number of blocks
    poisson_parameter = np.product(length_val)*tau_val

    prob_larger_than_0 = np.product((theta_val+(1-theta_val)*length_val)/length_val)
    K = np.random.poisson(poisson_parameter*prob_larger_than_0)

    # Initialize positions of blocks
    position_val = np.zeros((K, len(length_val), 2), dtype = int) # block number, block dimension, initial or termination position

    for dd in range(len(length_val)):
        dd_p = np.concatenate(([1], np.ones(length_val[dd]-1)*(1-theta_val[dd])))
        position_val[:, dd, 0] = np.random.choice(length_val[dd], size = K, replace = True, p = dd_p/sum(dd_p))+1

        # L_star = length_val[dd]-position_val[:, dd, 0] + 1

        side_length_pseudo = np.random.geometric(p = (1-theta_val[dd]), size = K)
        # side_length_pseudo[side_length_pseudo>=L_star] = L_star

        pseudo_termination_postion = position_val[:, dd, 0] + side_length_pseudo
        pseudo_termination_postion[pseudo_termination_postion>length_val[dd]] = length_val[dd]

        position_val[:, dd, 1] = pseudo_termination_postion

    return position_val

'''for i in range(K):
    print(position_val[i,:,0].T)
    print(position_val[i,:,1].T)
    print((position_val[i,0,1] - position_val[i,0,0]) * (position_val[i,1,1] - position_val[i,1,0]))
    print("------------------------")'''


