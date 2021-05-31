import unittest

import tskit
import msprime_pedigrees

import numpy as np


class TestSimulateFullTrios(unittest.TestCase):
    """
    Tests for the simulate_full_trios function.
    """

    def test_generations(self):
        for generations in range(4):
            tb = tskit.TableCollection(0).individuals
            msprime_pedigrees.simulate_full_trios(
                tb, generations=generations, families=1)
            n_individuals = 2**generations - 1
            self.assertEqual(len(tb), n_individuals)

    def test_families(self):
        for families in range(4):
            tb = tskit.TableCollection(0).individuals
            msprime_pedigrees.simulate_full_trios(
                tb, generations=1, families=families)
            n_individuals = families
            self.assertEqual(len(tb), n_individuals)

    def test_families_and_generations(self):
        for generations in range(1, 4):
            for families in range(1, 4):
                tb = tskit.TableCollection(0).individuals
                msprime_pedigrees.simulate_full_trios(
                    tb, generations=generations, families=families)
                n_individuals = (2**generations - 1) * families
                self.assertEqual(len(tb), n_individuals)


class TestMakeParents(unittest.TestCase):
    """
    Tests for the make_parents function.
    """
    DEFAULT_INDIVIDUALS_LIST = [[-1, -1, 1]]
    DEFAULT_KWARGS = {
        'iid': 0,
        'child_generation': 2,
    }

    def test_child_generation(self):
        for child_generation in range(4):
            kwargs = self.DEFAULT_KWARGS.copy()
            kwargs['individuals_list'] = self.DEFAULT_INDIVIDUALS_LIST.copy()
            kwargs['child_generation'] = child_generation
            individuals_list = msprime_pedigrees.make_parents(**kwargs)
            if child_generation <= 1:
                self.assertEqual(
                    individuals_list,
                    self.DEFAULT_INDIVIDUALS_LIST.copy())
            else:
                self.assertEqual(individuals_list[0], [1, 2, 1])
                self.assertEqual(len(individuals_list),
                                 (child_generation >= 1) * (2**child_generation - 1))
                self.assertEqual([x[2] for x in individuals_list[1:]], [
                                 1, 2] * (2**(child_generation - 1) - 1))

    def test_parent_ids(self):
        kwargs = self.DEFAULT_KWARGS.copy()
        kwargs['individuals_list'] = self.DEFAULT_INDIVIDUALS_LIST.copy()
        kwargs['child_generation'] = 2
        individuals_list = msprime_pedigrees.make_parents(**kwargs)
        self.assertEqual([x[:2] for x in individuals_list[1:]], [
                         [-1, -1]] * kwargs['child_generation'])

        kwargs = self.DEFAULT_KWARGS.copy()
        kwargs['individuals_list'] = self.DEFAULT_INDIVIDUALS_LIST.copy()
        kwargs['child_generation'] = 3
        individuals_list = msprime_pedigrees.make_parents(**kwargs)
        self.assertEqual([x[:2] for x in individuals_list[3:]], [
                         [-1, -1]] * (2**(kwargs['child_generation'] - 1)))

    def test_bad_iid(self):
        kwargs = self.DEFAULT_KWARGS.copy()
        kwargs['iid'] = 1
        kwargs['individuals_list'] = self.DEFAULT_INDIVIDUALS_LIST.copy()
        self.assertRaises(IndexError, msprime_pedigrees.make_parents, **kwargs)


class TestMakeChildren(unittest.TestCase):
    """
    Tests for the make_children function.
    """

    DEFAULT_INDIVIDUALS_LIST = [[-1, -1, 1], [-1, -1, 2]]
    DEFAULT_KWARGS = {
        'pat': 0,
        'mat': 1,
        'parent_generation': 1,
        'n_children': 1,
        'max_generations': 2,
        'n_children_prob': [1],
        'percent_w_partner': 0
    }

    def test_n_children(self):
        for n_children in range(4):
            kwargs = self.DEFAULT_KWARGS.copy()
            kwargs['individuals_list'] = self.DEFAULT_INDIVIDUALS_LIST.copy()
            kwargs['n_children'] = n_children
            individuals_list = msprime_pedigrees.make_children(**kwargs)
            if n_children == 0:
                self.assertEqual(
                    self.DEFAULT_INDIVIDUALS_LIST,
                    individuals_list)
            self.assertEqual(len(individuals_list), 2 + n_children)
            self.assertEqual([x[:2] for x in individuals_list[2:]],
                             [[0, 1]] * n_children)

    def test_generations(self):
        # if parent_generation < max_generation, child is created
        kwargs = self.DEFAULT_KWARGS.copy()
        kwargs['individuals_list'] = self.DEFAULT_INDIVIDUALS_LIST.copy()
        individuals_list = msprime_pedigrees.make_children(**kwargs)
        self.assertEqual(len(individuals_list), 3)
        self.assertEqual(individuals_list[2][:2], [0, 1])

        # if parent_generation = 1 = max_generation, no children are created
        kwargs = self.DEFAULT_KWARGS.copy()
        kwargs['individuals_list'] = self.DEFAULT_INDIVIDUALS_LIST.copy()
        kwargs['max_generations'] = 1
        individuals_list = msprime_pedigrees.make_children(**kwargs)
        self.assertEqual(
            self.DEFAULT_INDIVIDUALS_LIST,
            individuals_list)

        # if parent_generation = 2 = max_generation, no children are created
        kwargs = self.DEFAULT_KWARGS.copy()
        kwargs['individuals_list'] = self.DEFAULT_INDIVIDUALS_LIST.copy()
        kwargs['parent_generation'] = 2
        kwargs['max_generations'] = 2
        individuals_list = msprime_pedigrees.make_children(**kwargs)
        self.assertEqual(
            self.DEFAULT_INDIVIDUALS_LIST,
            individuals_list)


class TestMakePartner(unittest.TestCase):
    """
    Tests for the make_partner function.
    """
    pass


class TestMakeSiblings(unittest.TestCase):
    """
    Tests for the make_siblings function.
    """
    DEFAULT_INDIVIDUALS_LIST = [[-1, -1, 1]]
    DEFAULT_KWARGS = {
        'iid': 0,
        'generation': 1,
        'max_generations': 2,
        'n_children_prob': [1],
        'percent_w_partner': 0
    }

    def default_test_n_children_prob(self, n_children_prob):
        kwargs = self.DEFAULT_KWARGS.copy()
        kwargs['individuals_list'] = self.DEFAULT_INDIVIDUALS_LIST.copy()
        kwargs['n_children_prob'] = n_children_prob
        return msprime_pedigrees.make_siblings(**kwargs)

    def test_no_siblings(self):
        for n_children_prob in [[1], [0, 1]]:
            individuals_list = self.default_test_n_children_prob(
                n_children_prob)
            self.assertEqual(
                self.DEFAULT_INDIVIDUALS_LIST,
                individuals_list)

    def test_add_sibling(self):
        n_children_prob = [0, 0, 1]
        individuals_list = self.default_test_n_children_prob(n_children_prob)
        self.assertEqual(
            len(individuals_list),
            len(self.DEFAULT_INDIVIDUALS_LIST) + 1
        )

    def test_add_multiple_siblings(self):
        for n_additional_sibs in range(2, 4):
            n_children_prob = [0] * (n_additional_sibs + 1) + [1]
            individuals_list = self.default_test_n_children_prob(
                n_children_prob)
            self.assertEqual(
                len(individuals_list),
                len(self.DEFAULT_INDIVIDUALS_LIST) + n_additional_sibs
            )

    def test_bad_n_children_prob(self):
        # empty list of probabilities
        n_children_prob = []
        self.assertRaises(
            ValueError,
            self.default_test_n_children_prob,
            n_children_prob)

        # probabilities do not add up to 1
        n_children_prob = [0]
        self.assertRaises(
            ValueError,
            self.default_test_n_children_prob,
            n_children_prob)

    def test_n_children_prob(self):
        # equal chance of adding sib vs. not adding sib (may need to increase
        # replicates to avoid failing by chance)
        n_children_prob = [0, 0.5, 0.5]
        n_reps = 1000
        n_reps_w_additional_sib = 0
        np.random.seed(1)
        for rep in range(n_reps):
            individuals_list = self.default_test_n_children_prob(
                n_children_prob)
            if len(individuals_list) == len(self.DEFAULT_INDIVIDUALS_LIST) + 1:
                n_reps_w_additional_sib += 1
        self.assertAlmostEqual(
            n_reps_w_additional_sib,
            n_reps * n_children_prob[2],
            delta=n_reps * 0.05)
