import tskit

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