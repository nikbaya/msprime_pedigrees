

def plot_pedigree(individuals_list):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(list(range(len(individuals_list))))
    for iid, (pat, mat) in enumerate(individuals_list):
        if pat != -1 and mat != -1:
            # parents_node = f'{pat}-{mat}'
            # G.add_node(parents_node)
            G.add_edges_from(
                [(pat, mat, {'weight': 10}),
                 (pat, iid, {'weight': 1}),
                 (mat, iid, {'weight': 1})]
            )
            # G.add_edges_from(
            #     [(parents_node, iid, {'weight':1}),
            #      (pat, parents_node, {'weight':10}),
            #      (mat, parents_node, {'weight':10})])

    nx.draw(G, with_labels=True)


def main():
    # convert fam file to IndividualTable within TableCollection
    tc = tskit.TableCollection(0)
    msprime_pedigrees.fam_to_table(
        famfile='/Users/nbaya/Downloads/test.fam',
        tb=tc.individuals)
    print(tc.individuals)

    # convert list of lists to IndividualTable
    tc = tskit.TableCollection(0)
    individuals = [[2, 5, 6, 1], [0, 5, 6, 2], [1, 0, 3, 1]]
    list_to_table(individuals, tc.individuals)
    print(tc.individuals)

    # simulate more complex family tree

    percent_w_partner = 0.5
    max_generations = 4

    make_children(
        individuals_list,
        pat=0,
        mat=1,
        parent_generation=1,
        n_children=2)
