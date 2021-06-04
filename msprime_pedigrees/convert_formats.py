import numpy as np
import tskit


def array_to_table(individuals: np.ndarray, tc: tskit.TableCollection):
    '''
    Add individuals in array ``individuals``to IndividualTable in TableCollection ``tc``.

    :param np.ndarray individuals: Array of individuals. Each row should
        be a different individual. Each row must have five columns: FID, IID, PAT, MAT, SEX.
    :param tskit.IndividualTable tb: tskit ``IndividualTable``.
    '''
    if len(individuals) == 0:
        return tc
    id_map = {}  # dict for translating PLINK ID to tskit IndividualTable ID
    for tskit_id, (plink_fid, plink_iid, pat, mat,
                   sex) in enumerate(individuals):
        # need this to guarantee uniqueness of individual ID
        plink_id = str(plink_fid) +' '+ str(plink_iid) # include space between strings to ensure uniqueness
        if plink_id in id_map:
            raise ValueError('Duplicate PLINK ID')
        id_map[plink_id] = tskit_id
    id_map['0'] = '-1'

    tb = tc.individuals
    tb.metadata_schema = tskit.MetadataSchema(
        {
            "codec": "json",
            "type": "object",
            "properties": {
                    "plink_fid": {"type": "string"},
                    "plink_iid": {"type": "string"},
                    "sex": {"type": "integer"},
            },
            "required": ["plink_fid", "plink_iid", "sex"],
            "additionalProperties": True,
        }
    )
    for plink_fid, plink_iid, pat, mat, sex in individuals:
        sex = int(sex)
        if not (sex in range(3)):
            raise ValueError(
                'Sex must be one of the following integers: 0 (unknown), 1 (male), 2 (female)')
        metadata_dict = {
            'plink_fid': plink_fid,
            'plink_iid': plink_iid,
            'sex': sex}
        pat = plink_fid+' '+pat if pat != '0' else pat
        mat = plink_fid+' '+mat if mat != '0' else mat
        tb.add_row(
            parents=[id_map[pat],
                     id_map[mat]],
            metadata=metadata_dict
        )
    return tc


def fam_to_table(fname: str, tc: tskit.TableCollection):
    '''
    Convert fam file to tskit IndividualTable.

    Assumes fam file contains five columns: FID, IID, PAT, MAT, SEX
    FID is not used when converting to a table.
    '''
    individuals = np.loadtxt(
        fname,
        dtype=str,
        ndmin=2,  # read file as 2-D table
        usecols=(0, 1, 2, 3, 4)  # only keep FID, IID, PAT, MAT, SEX columns
    )  # requires same number of columns in each row, i.e. not ragged

    tc = array_to_table(individuals=individuals, tc=tc)
    return tc
