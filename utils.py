import numpy as np

def reset_start_and_target(x_in, cond, act_dim):
    if cond == {}:
        return x_in
    for key, val in cond.items():
        try:
            x_in[:, act_dim:, key] = val.clone()
        except Exception as e:
            print("Error in reset_start_and_target")
            print("x_in.shape: ", x_in.shape)
            print("act_dim: ", act_dim)
            print("key: ", key)
            print("val.shape: ", val.shape)
            print("x_in[:,act_dim:, key].shape: ", x_in[:, act_dim:, key].shape)
            print(e)
    return x_in


def limits_normalizer(x):
    '''
        Normalizes the input to the range [-1, 1]
    '''
    print("Normalizing data according to limits: ")
    print("x.min(): ", x.min())
    print("x.max(): ", x.max())
    return 2 * (x - x.min()) / (x.max() - x.min()) - 1

def limits_unnormalizer(x, min, max):
    '''
        Unormalizes the input from the range [-1, 1] to [min, max]
    '''
    if x.max() > 1 + 0.0001 or x.min() < -1 - 0.0001:
        print("Input is not normalized!", x.min(), x.max())
        x = np.clip(x, -1, 1)
    return 0.5 * (x + 1) * (max - min) + min