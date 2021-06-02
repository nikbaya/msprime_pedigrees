import unittest
import pytest
import tskit
import msprime_pedigrees
import numpy as np
import os


class TestListToTable(unittest.TestCase):
    """
    Tests for the list_to_table function.
    """

    def test_empty_list(self):
        tb = tskit.TableCollection(0).individuals
        individuals = []
        tb = msprime_pedigrees.list_to_table(individuals, tb)
        self.assertEqual(len(tb), 0)

    def test_bad_individuals(self):
        # not enough values to unpack
        tb = tskit.TableCollection(0).individuals
        individuals = [[] for _ in range(1)]
        self.assertRaises(IndexError, msprime_pedigrees.list_to_table,
                          individuals=individuals, tb=tb)

        # too many values to unpack
        tb = tskit.TableCollection(0).individuals
        individuals = [[5 * x + y for y in range(5)] + [0] for x in range(3)]
        self.assertRaises(ValueError, msprime_pedigrees.list_to_table,
                          individuals=individuals, tb=tb)

    def test_bad_sex_int(self):
        for bad_int in [-2, 3]:
            tb = tskit.TableCollection(0).individuals
            individuals = [[0, 0, 0, bad_int]]
            self.assertRaises(ValueError, msprime_pedigrees.list_to_table,
                              individuals=individuals, tb=tb)

    def test_sorted_individuals(self):
        # individuals are sorted in IndividualTable by original IID
        tb = tskit.TableCollection(0).individuals
        individuals = [[1, -1, -1, 1], [0, -1, -1, 2], [2, -1, -1, 2]]
        tb = msprime_pedigrees.list_to_table(individuals, tb)
        sorted_iids = sorted([x[0] for x in individuals])
        metadata_iids = [int(row.metadata.decode().split(' ')[0])
                             for row in tb]
        self.assertTrue(np.array_equal(sorted_iids, metadata_iids))

    def test_remap_parent_ids(self):
        # int parent IDs are remapped if the parents exist in the list
        tb = tskit.TableCollection(0).individuals
        individuals = [[11, 22, 33, 1], [22, -1, -1, 1], [33, -1, -1, 2]]
        tb = msprime_pedigrees.list_to_table(individuals, tb)
        self.assertTrue(np.array_equal(tb[0].parents, [1, 2]))

        # int parent IDs are set to missing (-1) if the parents do not exist in
        # the list
        tb = tskit.TableCollection(0).individuals
        individuals = [[0, 11, 22, 1], [1, 0, 0, 1], [2, 0, 0, 2]]
        tb = msprime_pedigrees.list_to_table(individuals, tb)
        self.assertTrue(np.array_equal(tb[0].parents, [-1, -1]))

        # str parent IDs are remapped if the parents exist in the list
        tb = tskit.TableCollection(0).individuals
        individuals = [['3', '1', '2', 1],
                       ['1', '0', '0', 1],
                       ['2', '0', '0', 2]]
        tb = msprime_pedigrees.list_to_table(individuals, tb)
        self.assertTrue(np.array_equal(tb[2].parents, [0, 1]))

        # str parent IDs are remapped if parents exist, regardless of int/str dtype
        tb = tskit.TableCollection(0).individuals
        individuals = [[0, '11', '22', 1], [11, '-1', '-1', 1], [22, '-1', '-1', 2]]
        tb = msprime_pedigrees.list_to_table(individuals, tb)
        self.assertTrue(np.array_equal(tb[0].parents, [1,2]))

class TestFamToTable(unittest.TestCase):
    """
    Tests for the fam_to_table function.
    """
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()

    def write_to_file(self, content):
        tmpdir = os.getcwd()
        fname = f'{tmpdir}/test.fam'
        f = open(fname, "w")
        f.write(content)
        f.close()
        return f

    def test_file_does_not_exist(self):
        tb = tskit.TableCollection(0).individuals
        self.assertRaises(OSError, msprime_pedigrees.fam_to_table, "", tb)
        
    def test_empty_file(self):
        f = self.write_to_file(content="")
        tb = tskit.TableCollection(0).individuals
        self.assertWarns(UserWarning, msprime_pedigrees.fam_to_table, famfile=f.name, tb=tb)
        tb = msprime_pedigrees.fam_to_table(f.name, tb)
        self.assertEqual(len(tb), 0)

    def test_insufficient_cols(self):
        for n_cols in range(1,5):
            entries = ['0']*n_cols # always fewer than the required 5 columns
            if n_cols >= 2:
                entries[1] = '1' # IID cannot be '0'
            f = self.write_to_file(content="\t".join(entries))
            tb = tskit.TableCollection(0).individuals
            self.assertRaises(IndexError, msprime_pedigrees.fam_to_table, famfile=f.name, tb=tb)
        
    def test_one_line_fam(self):
        entries = ['0']*6
        entries[1] = '1' # IID in fam file cannot be '0'
        f = self.write_to_file(content="\t".join(entries))
        tb = tskit.TableCollection(0).individuals
        tb = msprime_pedigrees.fam_to_table(f.name, tb)
        self.assertEqual(len(tb), 1)
        self.assertTrue(np.array_equal(tb[0].parents, [-1, -1]))
        self.assertTrue(np.array_equal(tb[0].metadata, bytes('1 -1 -1 0', 'utf-8')))
    
    def test_non_int_sex(self):
        entries = ['0']*6
        entries[1] = '1' # IID cannot be '0'
        entries[4] = 'F' # non-integer sex
        f = self.write_to_file(content="\t".join(entries))
        tb = tskit.TableCollection(0).individuals
        self.assertRaises(ValueError, msprime_pedigrees.fam_to_table, famfile=f.name, tb=tb)

    def test_bad_int_sex(self):
        entries = ['0']*6
        for bad_int in [-2, 3]:
            entries[4] = str(bad_int) # set sex to be a "bad" integer not in {0,1,2}
            f = self.write_to_file(content="\t".join(entries))
            tb = tskit.TableCollection(0).individuals
            self.assertRaises(ValueError, msprime_pedigrees.fam_to_table, famfile=f.name, tb=tb)
    
    def test_missing_phen_col(self):
        entries = ['0']*6 
        entries[1] = '1' # IID cannot be '0'
        
        f = self.write_to_file(content="\t".join(entries))
        tb = tskit.TableCollection(0).individuals
        tb = msprime_pedigrees.fam_to_table(f.name, tb)
        
        f_missing = self.write_to_file(content="\t".join(entries[:-1])) # ignore last column (PHEN column)
        tb_missing = tskit.TableCollection(0).individuals
        tb_missing = msprime_pedigrees.fam_to_table(f_missing.name, tb_missing)
        
        self.assertEqual(tb, tb_missing)
    
    def test_multiple_line_file(self):
        entries = [[0, 1, 0, 0, 0],
                   [0, 2, 0, 0, 0]]
        content = "\n".join(["\t".join(list(map(str, row))) for row in entries])
        f = self.write_to_file(content=content)
        tb = tskit.TableCollection(0).individuals
        tb = msprime_pedigrees.fam_to_table(f.name, tb)
        self.assertEqual(len(tb), 2)
        
    def test_duplicate_rows(self):
        entries = [[0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0]]
        content = "\n".join(["\t".join(list(map(str, row))) for row in entries])
        f = self.write_to_file(content=content)
        tb = tskit.TableCollection(0).individuals
        tb = msprime_pedigrees.fam_to_table(f.name, tb)
        self.assertEqual(len(tb), 2)
        
    def test_duplicate_row_as_parent(self):
        entries = [[0, 11, 0, 0, 0],
                   [0, 11, 0, 0, 0],
                   [0, 2, 11, 0, 0]]
        content = "\n".join(["\t".join(list(map(str, row))) for row in entries])
        f = self.write_to_file(content=content)
        tb = tskit.TableCollection(0).individuals
        tb = msprime_pedigrees.fam_to_table(f.name, tb)
        self.assertEqual(len(tb), 3)
        self.assertTrue(np.array_equal(tb[2].parents, [0, -1])) # parent ID is mapped to be the first instance of the duplicate individual
        
    def test_ragged_rows(self):
        entries = [[0, 1, 0, 0, 0],
                   [0, 2, '', 0, 0]]
        content = "\n".join(["\t".join(list(map(str, row))) for row in entries])
        f = self.write_to_file(content=content)
        tb = tskit.TableCollection(0).individuals
        self.assertRaises(IndexError, msprime_pedigrees.fam_to_table, famfile=f.name, tb=tb)
