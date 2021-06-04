import unittest
import pytest
import tskit
from msprime_pedigrees.convert_formats import array_to_table, fam_to_table
import numpy as np
import os


class TestArrayToTable(unittest.TestCase):
    """
    Tests for the array_to_table function.
    """

    def test_empty_list(self):
        tc = tskit.TableCollection(0)
        individuals = np.array([])
        tc = array_to_table(individuals, tc)
        self.assertEqual(len(tc.individuals), 0)

    def test_bad_individuals(self):
        # not enough values to unpack
        tc = tskit.TableCollection(0)
        individuals = np.array([[]])
        self.assertRaises(ValueError, array_to_table,
                          individuals=individuals, tc=tc)

        # too many values to unpack
        tc = tskit.TableCollection(0)
        individuals = np.array(['0', '1', '0', '0', '0', '0'])
        self.assertRaises(ValueError, array_to_table,
                          individuals=individuals, tc=tc)

    def test_bad_sex_value(self):
        # SEX integer not in {0,1,2}
        for bad_int in [-2, 3]:
            tc = tskit.TableCollection(0)
            individuals = np.array([['0', '1', '0', '0', str(bad_int)]])
            self.assertRaises(ValueError, array_to_table,
                              individuals=individuals, tc=tc)
        
        # invalid SEX string
        tc = tskit.TableCollection(0)
        individuals = np.array([['0', '1', '0', '0', 'F']])
        self.assertRaises(ValueError, array_to_table,
                          individuals=individuals, tc=tc)
        
        # empty string
        tc = tskit.TableCollection(0)
        individuals = np.array([['0', '1', '0', '0', '']])
        self.assertRaises(ValueError, array_to_table,
                          individuals=individuals, tc=tc)

    def test_single_individual(self):
        tc = tskit.TableCollection(0)
        individuals = np.array([['0', '1', '0', '0', '0']])
        tc = array_to_table(individuals, tc)
        tb = tc.individuals
        self.assertEqual(len(tb), 1)
        self.assertTrue(np.array_equal(tb[0].parents, [-1, -1]))
        self.assertEqual(tb[0].metadata['plink_fid'], '0')
        self.assertEqual(tb[0].metadata['plink_iid'], '1')
        self.assertEqual(tb[0].metadata['sex'], 0)

    def test_individuals_unsorted(self):
        tc = tskit.TableCollection(0)
        individuals = np.array([['0', '3', '0', '0', '0'],
                                ['0', '1', '0', '0', '0'],
                                ['0', '2', '0', '0', '0']])
        tc = array_to_table(individuals, tc)
        tb = tc.individuals
        plink_iids = individuals[:, 1]
        tb_plink_iids = [row.metadata['plink_iid'] for row in tb]
        self.assertTrue(np.array_equal(plink_iids, tb_plink_iids))

    def test_duplicate_within_family_iid(self):
        tc = tskit.TableCollection(0)
        individuals = np.array([['0', '1', '0', '0', '0'],
                                ['1', '1', '0', '0', '0']])
        tc = array_to_table(individuals, tc)
        tb = tc.individuals
        self.assertEqual(len(tb), 2)
        self.assertEqual(tb[0].metadata['plink_fid'], '0')
        self.assertEqual(tb[0].metadata['plink_iid'], '1')
        self.assertEqual(tb[1].metadata['plink_fid'], '1')
        self.assertEqual(tb[1].metadata['plink_iid'], '1')

    def test_single_family_map_parent_ids(self):
        # PAT is mapped if the individual exists in the dataset
        tc = tskit.TableCollection(0)
        individuals = np.array([['0', '1', '2', '0', '0'],
                                ['0', '2', '0', '0', '0']])
        tc = array_to_table(individuals, tc)
        self.assertTrue(np.array_equal(tc.individuals[0].parents, [1, -1]))
        
        # MAT is mapped if the individual exists in the dataset
        tc = tskit.TableCollection(0)
        individuals = np.array([['0', '1', '0', '2', '0'],
                                ['0', '2', '0', '0', '0']])
        tc = array_to_table(individuals, tc)
        self.assertTrue(np.array_equal(tc.individuals[0].parents, [-1, 1]))
        
        # both parent IDs are remapped if the both parents exist in the dataset
        tc = tskit.TableCollection(0)
        individuals = np.array([['0', '1', '2', '3', '0'],
                                ['0', '2', '0', '0', '0'],
                                ['0', '3', '0', '0', '0']])
        tc = array_to_table(individuals, tc)
        self.assertTrue(np.array_equal(tc.individuals[0].parents, [1, 2]))
        
        # KeyError raised if at least one parent (PAT) does not exist in dataset
        tc = tskit.TableCollection(0)
        individuals = np.array([['0', '1', '2', '3', '0'],
                                ['0', '3', '0', '0', '0']])
        self.assertRaises(KeyError, array_to_table, individuals=individuals, tc=tc)

        # KeyError raised if at least one parent (MAT) does not exist in dataset
        tc = tskit.TableCollection(0)
        individuals = np.array([['0', '1', '2', '3', '0'],
                                ['0', '2', '0', '0', '0']])
        self.assertRaises(KeyError, array_to_table, individuals=individuals, tc=tc)
        
    def test_multiple_family_map_parent_ids(self):
        # parents mapped correctly when the same parent ID is used in different families
        tc = tskit.TableCollection(0)
        individuals = np.array([['0', '1', '2', '3', '0'],
                                ['0', '2', '0', '0', '0'],
                                ['0', '3', '0', '0', '0'],
                                ['1', '1', '2', '3', '0'],
                                ['1', '2', '0', '0', '0'],
                                ['1', '3', '0', '0', '0']])
        tc = array_to_table(individuals, tc)
        tb = tc.individuals
        self.assertTrue(np.array_equal(tb[0].parents, [1, 2]))
        self.assertTrue(np.array_equal(tb[3].parents, [4, 5]))

        # KeyError raised even if parent ID matches, but FID does not
        individuals = np.array([['0', '1', '2', '3', '0'],
                                ['0', '2', '0', '0', '0'],
                                ['0', '3', '0', '0', '0'],
                                ['1', '1', '2', '3', '0'],
                                ['1', '2', '0', '0', '0']])
        self.assertRaises(KeyError, array_to_table, individuals=individuals, tc=tc)
        
    def test_grandparents(self):
        tc = tskit.TableCollection(0)
        individuals = np.array([['0', '1', '2', '3', '0'],
                                ['0', '2', '4', '5', '0'],
                                ['0', '3', '6', '7', '0'],
                                ['0', '4', '0', '0', '0'],
                                ['0', '5', '0', '0', '0'],
                                ['0', '6', '0', '0', '0'],
                                ['0', '7', '0', '0', '0']])
        tc = array_to_table(individuals, tc)
        self.assertTrue(np.array_equal(tc.individuals[0].parents, [1, 2]))
        self.assertTrue(np.array_equal(tc.individuals[1].parents, [3, 4]))
        self.assertTrue(np.array_equal(tc.individuals[2].parents, [5, 6]))

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
        tc = tskit.TableCollection(0)
        self.assertRaises(OSError, fam_to_table, "", tc)

    def test_empty_file(self):
        f = self.write_to_file(content="")
        tc = tskit.TableCollection(0)
        self.assertWarns(UserWarning, fam_to_table, fname=f.name, tc=tc)
        tc = fam_to_table(f.name, tc)
        self.assertEqual(len(tc.individuals), 0)

    def test_insufficient_cols(self):
        for n_cols in range(1, 5):
            # always fewer than the required 5 columns
            entries = ['0'] * n_cols
            if n_cols >= 2:
                entries[1] = '1'  # IID cannot be '0'
            f = self.write_to_file(content="\t".join(entries))
            tc = tskit.TableCollection(0)
            self.assertRaises( IndexError, fam_to_table, fname=f.name, tc=tc)

    def test_single_line_file(self):
        entries = ['0'] * 6
        entries[1] = '1'  # IID in fam file cannot be '0'
        f = self.write_to_file(content="\t".join(entries))
        tc = tskit.TableCollection(0)
        tc = fam_to_table(f.name, tc)

        self.assertEqual(len(tc.individuals), 1)
        self.assertTrue(np.array_equal(tc.individuals[0].parents, [-1, -1]))
        self.assertEqual(tc.individuals[0].metadata['plink_fid'], '0')
        self.assertEqual(tc.individuals[0].metadata['plink_iid'], '1')
        self.assertEqual(tc.individuals[0].metadata['sex'], 0)

    def test_multiple_line_file(self):
        entries = [[0, 1, 0, 0, 0],
                   [0, 2, 0, 0, 0]]
        content = "\n".join(["\t".join(list(map(str, row)))
                             for row in entries])
        f = self.write_to_file(content=content)
        tc = tskit.TableCollection(0)
        tc = fam_to_table(f.name, tc)
        self.assertEqual(len(tc.individuals), 2)
        for idx in range(2):
            self.assertTrue(np.array_equal(tc.individuals[idx].parents, [-1, -1]))
            self.assertEqual(tc.individuals[idx].metadata['plink_fid'], '0')
            self.assertEqual(tc.individuals[idx].metadata['plink_iid'], str(entries[idx][1]))
            self.assertEqual(tc.individuals[idx].metadata['sex'], 0)


    def test_missing_phen_col(self):
        entries = ['0'] * 6
        entries[1] = '1'  # IID cannot be '0'

        f = self.write_to_file(content="\t".join(entries))
        tc = tskit.TableCollection(0)
        tc = fam_to_table(f.name, tc)

        f_missing = self.write_to_file(content="\t".join(
            entries[:-1]))  # ignore last column (PHEN column)
        tc_missing = tskit.TableCollection(0)
        tc_missing = fam_to_table(f_missing.name, tc_missing)

        self.assertEqual(tc, tc_missing)

    def test_duplicate_rows(self):
        entries = [[0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0]]
        content = "\n".join(["\t".join(list(map(str, row)))
                             for row in entries])
        f = self.write_to_file(content=content)
        tc = tskit.TableCollection(0)
        self.assertRaises(ValueError, fam_to_table, fname=f.name, tc=tc)

    def test_space_delimited(self):
        entries = ['0'] * 5
        entries[1] = '1'  # IID cannot be '0'
        f = self.write_to_file(content=" ".join(entries))
        tc = tskit.TableCollection(0)
        tc = fam_to_table(f.name, tc)
        self.assertTrue(np.array_equal(tc.individuals[0].parents, [-1, -1]))
        self.assertEqual(tc.individuals[0].metadata['plink_fid'], '0')
        self.assertEqual(tc.individuals[0].metadata['plink_iid'], '1')
        self.assertEqual(tc.individuals[0].metadata['sex'], 0)
