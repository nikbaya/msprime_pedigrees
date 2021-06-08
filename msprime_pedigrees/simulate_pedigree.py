import tskit
import numpy as np
# import networkx as nx

# Thousands of families with 0 children, 1 child, etc., adapted from
# https://www.statista.com/statistics/183790/number-of-families-in-the-us-by-number-of-children/
N_CHILDREN_CTS = [50, 14, 13, 5, 2, 1]
N_CHILDREN_PROB = np.asarray(N_CHILDREN_CTS) / sum(N_CHILDREN_CTS)

PERCENT_W_PARTNER = 0.5

def choose_sex():
    sex = 1 + np.random.binomial(n=1, p=0.5)  # 1 = male, 2 = female
    return sex

def choose_n_children(n_children_prob):
    return np.random.choice(a=len(n_children_prob), p=n_children_prob)

def simulate_pedigree_from_founders():
    def is_valid_pair(pat_anc, mat_anc, mat_idx):
        common_anc = set(pat_anc).intersection(mat_anc)
        return len(common_anc) == 0
    np.random.seed(5)

    n_founders = 10000
    n_males = int(n_founders / 2)

    # n_children_prob = [0.2, 0.2, 0.2, 0.2, 0.2]
    n_children_prob = [0, 0, 0, 1]

    n_total = n_founders

    # set up
    curr_gen = [[], []]
    sex = 1
    for pat in range(n_males):
        mat = n_males + pat
        n_children = choose_n_children(n_children_prob=n_children_prob)
        for _ in range(n_children):
            curr_gen[sex] += [[[pat, mat]]]
            sex = (sex + 1) % 2

    n_generations = 10
    max_gen_idx = 2  # max_gen_idx=2 -> 3 generations back
    for _ in range(n_generations - 1):
        n_males = len(curr_gen[0])
        n_females = len(curr_gen[1])
        # print(n_males, n_females)
        next_gen = [[], []]
        avail_mat = np.random.permutation(n_females)
        for pat_idx in range(n_males):
            pat = pat_idx
            pat_anc = curr_gen[0][pat_idx][-1]
            for mat_idx in avail_mat:
                mat_anc = curr_gen[1][mat_idx][-1]
                if is_valid_pair(pat_anc, mat_anc, mat_idx):
                    break
            if is_valid_pair(pat_anc, mat_anc, mat_idx) and len(
                    avail_mat) > 0:  # unrelated male/female pair has been made
                avail_mat = avail_mat[avail_mat != mat_idx]
                n_children = choose_n_children(n_children_prob=n_children_prob)
                n_total += n_children
                mat = mat_idx + n_males
                anc = [[pat, mat]]
                # number of generations for paternal and maternal ancestors
                # should be the same
                for gen_idx in range(
                        min(max_gen_idx, len(curr_gen[0][pat_idx]))):
                    pat_anc = curr_gen[0][pat_idx][gen_idx]
                    mat_anc = curr_gen[1][mat_idx][gen_idx]
                    anc += [pat_anc + mat_anc]
                for _ in range(n_children):
                    next_gen[sex] += [anc]
                    sex = (sex + 1) % 2
            # else:
            #     print(f'no match for pat={pat}')
        # print(next_gen[0])
        # print(next_gen[1])
        curr_gen = next_gen

        print(f'current gen: {len(curr_gen[0])+len(curr_gen[1])}')
    print(f'n_total: {n_total}')
