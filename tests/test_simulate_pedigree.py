# MIT License
#
# Copyright (c) 2021 Tskit Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest

import tskit
from msprime_pedigrees.simulate_pedigree import simulate_pedigree

import numpy as np


class TestPedigreeSimulation():
    """
    Tests for simple pedigree simulator
    """
    def simple_sim(
            self,
            n_founders=2, 
            n_children_prob=[0, 0, 1],
            n_generations = 2,
            random_sex=True):
        tc=tskit.TableCollection(0)
        simulate_pedigree(
            tc=tc,
            n_founders=n_founders, 
            n_children_prob=n_children_prob,
            n_generations=n_generations,
            random_sex=random_sex)
        return tc.individuals
    
    def test_one_generation_no_children(self):
        n_founders = 8
        tb = self.simple_sim(
            n_founders=n_founders,
            n_generations=1)
        assert len(tb)==n_founders
        
        # check that all parents of founders are missing
        assert all([np.array_equal(row.parents, [-1,-1]) for row in tb])

        n_males = len([row for row in tb if row.metadata['sex']==1])
        n_females = len([row for row in tb if row.metadata['sex']==2])
        assert n_males==n_females==n_founders/2
        
    def test_one_trio(self):
        tb = self.simple_sim(
            n_founders=2,
            n_children_prob=[0,1],
            n_generations=2)
        assert len(tb)==3
        assert np.array_equal(tb[2].parents, [0,1])
        assert all([np.array_equal(row.parents, [-1,-1]) for row in tb[:2]])
        
    def test_grandparents(self):
        tb = self.simple_sim(
            n_founders=4,
            n_children_prob=[0,1],
            n_generations=3,
            random_sex=False)
        assert len(tb)==7
        assert {row.parents[0] for row in tb[4:6]}=={0,1}
        assert {row.parents[1] for row in tb[4:6]}=={2,3}
        assert all([np.array_equal(row.parents, [-1,-1]) for row in tb[:4]])        

    def test_insufficient_founders(self):
        with pytest.raises(Exception):
            self.simple_sim(
                n_founders=1,
                n_children_prob=[0,1])
        with pytest.raises(Exception):
            self.simple_sim(
                n_founders=3,
                n_children_prob=[0,1],
                n_generations=3)
    
    @pytest.mark.parametrize("n_children", range(5))
    def test_nonrandom_child_prob(self, n_children):
        tb = self.simple_sim(
            n_founders=2,
            n_children_prob=[0]*n_children+[1],
            n_generations=2)
        assert len(tb)==2+n_children
        
    @pytest.mark.parametrize("n_children", [[0,0.5,0.5]])
    def test_expected_n_children(self, n_children):
        pass
    
    def test_bad_n_children_prob(self):
        with pytest.raises(ValueError):
            self.simple_sim(n_children_prob=[2])
        with pytest.raises(ValueError):
            self.simple_sim(n_children_prob=[1,1])
        