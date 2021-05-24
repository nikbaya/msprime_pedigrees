import tskit
import numpy as np
import networkx as nx

def simulate_full_trios(
    tb : tskit.IndividualTable, 
    generations : int = 2, 
    families : int = 1
):
    get_n_per_gen = lambda generation: 2**(generation)
    get_total_n = lambda generations: 2**(generations)-1
    for fam in range(families):
        start_idx = tb.num_rows
        for generation in range(generations, 0, -1):
            for idx in range(get_n_per_gen(generation-1)):
                first_parent_idx = start_idx + get_total_n(generations) - get_total_n(generation+1) + 2*idx
                parents = [first_parent_idx , first_parent_idx+1] if generation < generations else [-1,-1]
                tb.add_row(location=[generation], parents=parents)
                
def simulate_random_families( tb : tskit.IndividualTable, max_generations : int ,
                             n_children_prob, percent_w_spouse):
    def choose_sex():
        sex = 1+np.random.binomial(n=1, p=0.5) # 1 = male, 2 = female
        return sex
    
    def make_parents( individuals_list, iid, child_generation):
        if child_generation-1 >= 1:
            # print(f'make parents for {iid}-{child_generation}')
            pat, mat = [len(individuals_list)+x for x in range(2)]
            individuals_list[iid] = [pat, mat]
            individuals_list += [[-1,-1], [-1,-1]] # add rows with unknown parents for the two parents of iid
            # print(individuals_list)
            child_generation -= 1
            for iid in [pat, mat]:
                individuals_list = make_parents( individuals_list, iid=iid, child_generation = child_generation)
        return individuals_list
    
    def make_children( individuals_list, pat, mat, parent_generation, n_children ):
        if parent_generation+1 <= max_generations and n_children>0:
            # print(f'make children for {pat}/{mat}-{parent_generation}')
            first_child_iid = len(individuals_list)
            individuals_list += [[pat, mat]]*n_children
            # print(individuals_list)
            for iid in [first_child_iid+x for x in range(n_children)]:
                sex = choose_sex()
                individuals_list = make_partner( individuals_list, iid = iid, sex = sex, generation = parent_generation+1)
        return individuals_list
    
    def make_partner( individuals_list, iid, sex, generation ):
        n_children = np.random.choice(a=len(n_children_prob), p=n_children_prob)
        if np.random.binomial(n=1, p=percent_w_spouse) and n_children>0 and generation+1 <= max_generations:
            # print(f'make spouse for {iid}-{generation}, {"M" if sex==1 else "F"}')
            spouse_iid = len(individuals_list)
            individuals_list += [[-1,-1]]
            # print(individuals_list)
            individuals_list = make_parents( individuals_list, iid=spouse_iid, child_generation=generation )
            pat = iid if sex==1 else spouse_iid
            mat = iid if sex==2 else spouse_iid
            individuals_list = make_children( individuals_list, pat=pat, mat=mat, parent_generation=generation, n_children=n_children)
            individuals_list = make_siblings( individuals_list, iid=spouse_iid, generation=generation)
        return individuals_list
        
    def make_siblings( individuals_list, iid, generation ):
        n_children = np.random.choice(a=len(n_children_prob), p=n_children_prob)
        n_siblings = n_children-1
        if n_siblings > 0:
            # print(f'make sibs for {iid}-{generation}')
            first_sibling_iid = len(individuals_list)
            pat, mat = individuals_list[iid]
            individuals_list += [[pat, mat]]*n_siblings
            # print(individuals_list)
            for iid in [first_sibling_iid+x for x in range(n_siblings)]:
                sex = choose_sex()
                if np.random.binomial(n=1, p=percent_w_spouse):
                    individuals_list = make_partner( individuals_list, iid=iid, sex=sex, generation=generation)
        return individuals_list
    
    
def list_to_table( individuals, tb : tskit.IndividualTable ):
    '''
    Add individuals in array or list of lists ``individuals``to ``tb``.
    
    :param individuals: Array of individuals, or list of lists. Each row should 
        be a different individual. Each row should have three entries: IID, PID1, PID2.
    :param tskit.IndividualTable tb: tskit ``IndividualTable``.
    '''
    individuals = sorted(individuals, key = lambda x: x[0]) # sort by first entry
    iid_to_idx = {iid: idx for idx, (iid, _, _, _) in enumerate(individuals)}
    for idx, (iid, pid1, pid2, sex) in enumerate(individuals):
        assert sex in range(3), 'Sex must be one of the following integers: 0 (unknown), 1 (male), 2 (female)'
        tb.add_row(
            parents=[ iid_to_idx.setdefault(pid1,-1), iid_to_idx.setdefault(pid2,-1) ],
            metadata=bytes(str(int(sex)), 'utf-8') # bytes('-'.join(map(str, [iid,pid1,pid2])), 'utf-8')
        )

def fam_to_table(famfile : str, tb : tskit.IndividualTable ):
    '''
    Convert fam file to tskit IndividualTable.
    '''
    fam = np.loadtxt(famfile, dtype=str) # requires same number of columns in each row, i.e. not ragged
    individuals = fam[:,1:4]
    list_to_table( individuals=individuals, tb=tb)

def main():
    # convert fam file to IndividualTable within TableCollection
    tc = tskit.TableCollection(0)
    fam_to_table(famfile='/Users/nbaya/Downloads/test.fam', tb=tc.individuals)
    print(tc.individuals)    

    # convert list of lists to IndividualTable
    tc = tskit.TableCollection(0)
    individuals = [[2,5,6,1],[0,5,6,2],[1,0,3,1]]
    list_to_table(individuals, tc.individuals)
    print(tc.individuals)

    # simulate full trios
    tc = tskit.TableCollection(0)
    simulate_full_trios(tc.individuals, generations=3, families=3)
    print(tc.individuals)
    
    # simulate more complex family tree
    n_children_cts = [50, 14, 13, 5, 2, 1] # thousands of families with 0 children, 1 child, etc., adapted from from https://www.statista.com/statistics/183790/number-of-families-in-the-us-by-number-of-children/
    n_children_prob = np.asarray(n_children_cts)/sum(n_children_cts)
    percent_w_spouse = 0.5
    max_generations = 4
    individuals_list = [[-1,-1,1],[-1,-1,2]]
    make_children( individuals_list, pat=0, mat=1, parent_generation=1 , n_children = 2)
         
    G = nx.Graph()
    G.add_nodes_from(list(range(len(individuals_list))))
    for iid, (pat,mat) in enumerate(individuals_list):
        if pat!=-1 and mat!=-1:
            # parents_node = f'{pat}-{mat}'
            # G.add_node(parents_node)
            G.add_edges_from(
                [(pat, mat, {'weight':10}),
                 (pat, iid, {'weight':1}),
                 (mat, iid, {'weight':1})]
            )
            # G.add_edges_from(
            #     [(parents_node, iid, {'weight':1}), 
            #      (pat, parents_node, {'weight':10}), 
            #      (mat, parents_node, {'weight':10})])
        
    nx.draw(G, with_labels=True)

    
    