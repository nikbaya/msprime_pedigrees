import tskit
import numpy as np
# import networkx as nx

# Thousands of families with 0 children, 1 child, etc., adapted from
# https://www.statista.com/statistics/183790/number-of-families-in-the-us-by-number-of-children/
N_CHILDREN_CTS = [50, 14, 13, 5, 2, 1]
N_CHILDREN_PROB = np.asarray(N_CHILDREN_CTS) / sum(N_CHILDREN_CTS)

PERCENT_W_PARTNER = 0.5

def simulate_pedigree(
        tc: tskit.TableCollection,
        n_founders = 8, 
        n_children_prob = [0, 0, 2],
        n_generations = 3,
        random_sex=True):

    def choose_sex(random=True, sex=None):
        # male = 1, female = 2
        if random:
            sex = np.random.binomial(n=1, p=0.5) + 1
        else: # non-random: alternate male female to ensure equal balance
            sex = (sex % 2) + 1 
        return sex
    
    def choose_n_children(n_children_prob, n_families=1):
        return np.random.choice(
            a=len(n_children_prob), 
            size=n_families, 
            p=n_children_prob)
    
    tb = tc.individuals
    tb.metadata_schema = tskit.MetadataSchema(
        {
            "codec": "json",
            "type": "object",
            "properties": {
                "sex": {"type": "integer"},
            },
            "required": ["sex"],
            "additionalProperties": True,
        }
    )
   
    if random_sex:
        sex=None
    else:
        sex = 1 # male=1, female=2
    
    # first generation
    n_males = int(n_founders / 2)
    curr_gen = [range(n_males), range(n_males, n_founders)]
    for ind in range(n_founders):
        tb.add_row(
            parents=[-1, -1],
            metadata={'sex': (ind>=n_males)+1}
        )

    # next generations            
    for gen_idx in range(2, n_generations+1):
        next_gen = [[], []]
        avail_pat = np.random.permutation(curr_gen[0])
        avail_mat = np.random.permutation(curr_gen[1])
        n_pairs = min(len(curr_gen[0]), len(curr_gen[1]))
        if n_pairs==0 and n_children_prob[0]!=1:
            raise Exception(f"Not enough people to make children in generation {gen_idx}")
        pairs = zip(avail_pat[:n_pairs], avail_mat[:n_pairs])
        n_children_per_pair = choose_n_children(
            n_children_prob=n_children_prob,
            n_families=n_pairs
        )
        for (pat, mat), n_children in zip(pairs, n_children_per_pair):
            for _ in range(n_children):
                sex = choose_sex(
                    random=random_sex, 
                    sex=sex)
                next_gen[sex-1] += [len(tb)]
                tb.add_row(
                    parents=[pat, mat],
                    metadata={'sex': sex}
                )
        curr_gen = next_gen