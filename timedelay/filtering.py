import numpy as np

def filter_pairs(pair_ids, system_type="double"):
    quad_pair_ids = pair_ids[pair_ids % 1.0 == 0.5] # Finde file A of a quadratic system
    quad_pair_ids = np.append(quad_pair_ids, quad_pair_ids - 0.5) # Finds file B of the same quadratic system
    if system_type == "double":
        return np.setdiff1d(pair_ids, quad_pair_ids) # Removes all quadratic systems from pair_ids
    elif system_type == "quad":
        return quad_pair_ids # Chooses all quadratic systems from pair_ids
    return  pair_ids
