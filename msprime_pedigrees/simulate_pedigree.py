import tskit
import numpy as np
# import networkx as nx

# thousands of families with 0 children, 1 child, etc., adapted from
# https://www.statista.com/statistics/183790/number-of-families-in-the-us-by-number-of-children/
N_CHILDREN_CTS = [50, 14, 13, 5, 2, 1]
N_CHILDREN_PROB = np.asarray(N_CHILDREN_CTS) / sum(N_CHILDREN_CTS)

PERCENT_W_PARTNER = 0.5


def simulate_full_trios(
    tb: tskit.IndividualTable,
    generations: int = 2,
    families: int = 1
):
    def get_n_per_gen(generation): return 2**(generation)
    def get_total_n(generations): return 2**(generations) - 1
    for fam in range(families):
        start_idx = tb.num_rows
        for generation in range(generations, 0, -1):
            for idx in range(get_n_per_gen(generation - 1)):
                first_parent_idx = start_idx + \
                    get_total_n(generations) - \
                    get_total_n(generation + 1) + 2 * idx
                parents = [first_parent_idx, first_parent_idx +
                           1] if generation < generations else [-1, -1]
                tb.add_row(location=[generation], parents=parents)


def choose_sex():
    sex = 1 + np.random.binomial(n=1, p=0.5)  # 1 = male, 2 = female
    return sex


def make_parents(individuals_list, iid, child_generation):
    if child_generation - 1 >= 1:
        # print(f'make parents for {iid}-{child_generation}')
        pat, mat = [len(individuals_list) + x for x in range(2)]
        sex = individuals_list[iid][2]
        individuals_list[iid] = [pat, mat, sex]
        # add rows with unknown parents for the two parents of iid
        individuals_list += [[-1, -1, 1], [-1, -1, 2]]
        # print(individuals_list)
        child_generation -= 1
        for iid in [pat, mat]:
            individuals_list = make_parents(
                individuals_list=individuals_list,
                iid=iid,
                child_generation=child_generation
            )
    return individuals_list


def make_children(individuals_list, pat, mat, parent_generation, n_children,
                  max_generations, n_children_prob=N_CHILDREN_PROB, percent_w_partner=PERCENT_W_PARTNER):
    if parent_generation + 1 <= max_generations and n_children > 0:
        # print(f'make children for {pat}/{mat}-{parent_generation}')
        first_child_iid = len(individuals_list)
        individuals_list += [[pat, mat, choose_sex()]] * n_children
        # print(individuals_list)
        for iid in [first_child_iid + x for x in range(n_children)]:
            _, _, sex = individuals_list[iid]
            individuals_list = make_partner(
                individuals_list=individuals_list,
                iid=iid,
                sex=sex,
                generation=parent_generation + 1,
                max_generations=max_generations,
                n_children_prob=n_children_prob,
                percent_w_partner=percent_w_partner
            )
    return individuals_list


def make_partner(individuals_list, iid, sex, generation, max_generations,
                 n_children_prob=N_CHILDREN_PROB, percent_w_partner=PERCENT_W_PARTNER):
    n_children = np.random.choice(a=len(n_children_prob), p=n_children_prob)
    if (np.random.binomial(n=1, p=percent_w_partner)
        and n_children > 0
            and generation + 1 <= max_generations):
        # print(f'make partner for {iid}-{generation}, {"M" if sex==1 else "F"}')
        partner_iid = len(individuals_list)
        individuals_list += [[-1, -1]]
        # print(individuals_list)
        individuals_list = make_parents(
            individuals_list=individuals_list,
            iid=partner_iid,
            child_generation=generation
        )
        pat = iid if sex == 1 else partner_iid
        mat = iid if sex == 2 else partner_iid
        individuals_list = make_children(
            individuals_list=individuals_list,
            pat=pat,
            mat=mat,
            parent_generation=generation,
            n_children=n_children
        )
        individuals_list = make_siblings(
            individuals_list=individuals_list,
            iid=partner_iid,
            generation=generation,
            n_children_prob=n_children_prob,
            percent_w_partner=percent_w_partner
        )
    return individuals_list


def make_siblings(individuals_list, iid, generation, max_generations,
                  n_children_prob=N_CHILDREN_PROB, percent_w_partner=PERCENT_W_PARTNER):
    n_children = np.random.choice(a=len(n_children_prob), p=n_children_prob)
    n_siblings = n_children - 1
    if n_siblings > 0:
        # print(f'make sibs for {iid}-{generation}')
        first_sibling_iid = len(individuals_list)
        pat, mat, _ = individuals_list[iid]
        individuals_list += [[pat, mat, choose_sex()]] * n_siblings
        # print(individuals_list)
        for iid in [first_sibling_iid + x for x in range(n_siblings)]:
            _, _, sex = individuals_list[iid]
            individuals_list = make_partner(
                individuals_list=individuals_list,
                iid=iid,
                sex=sex,
                generation=generation,
                max_generations=max_generations,
                n_children_prob=n_children_prob,
                percent_w_partner=percent_w_partner
            )
    return individuals_list
