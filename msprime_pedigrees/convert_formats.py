import numpy as np
import tskit


def list_to_table(individuals, tb: tskit.IndividualTable):
    '''
    Add individuals in array or list of lists ``individuals``to ``tb``.

    :param individuals: Array of individuals, or list of lists. Each row should
        be a different individual. Each row should have at least three columns: IID, PAT, MAT.
        Another column, SEX, may also be included.
    :param tskit.IndividualTable tb: tskit ``IndividualTable``.
    '''
    if len(individuals)==0:
        return tb
    individuals = np.array(individuals)
    individuals = individuals[individuals[:,0].argsort(), :] # sort rows by first column (IID)
    iid_to_idx = {} # dict for translating old iid to new idx
    for idx, row in enumerate(individuals):
        iid = row[0]
        if iid not in iid_to_idx: # need this check to avoid overwriting existing values
            iid_to_idx[iid] = idx        
        # individuals[idx] = np.append(row, [-1]*(4-len(row))).astype(int) # pad with -1
    for idx, (iid, pat, mat, sex)in enumerate(individuals):
        sex = 0 if int(sex)==-1 else int(sex) # unknown value (-1) for sex is translated to be coded as 0
        if not (sex in range(3)):
            raise ValueError(
                'Sex must be one of the following integers: 0 (unknown), 1 (male), 2 (female)')
        tb.add_row(
            parents=[iid_to_idx.setdefault(pat, -1),
                     iid_to_idx.setdefault(mat, -1)],
            metadata=bytes(' '.join(map(str, [iid, pat, mat, sex])), 'utf-8')
        )
    return tb


def fam_to_table(famfile: str, tb: tskit.IndividualTable):
    '''
    Convert fam file to tskit IndividualTable.
    
    Assumes fam file contains five columns: FID, IID, PAT, MAT, SEX
    FID is not used when converting to a table.
    '''
    convert_missing_pids = lambda s: s.replace(b'0', b'-1') # convert missing parent IDs from "0" to "-1"
    individuals = np.loadtxt(
        famfile,
        converters = {2: convert_missing_pids,
                      3: convert_missing_pids},
        dtype=str,
        ndmin=2, # read file as 2-D table
        usecols = (1,2,3,4) # only keep IID, PAT, MAT, SEX columns
    )  # requires same number of columns in each row, i.e. not ragged

    tb = list_to_table(individuals=individuals, tb=tb)
    return tb
