
import unittest
import math

import torch
import numpy as np

from fairseq.incrcorr import IncrCorr

# Correlation calculation "by hand" to use
# for comparison.
# Not needed anymore, since numpy produces
# the same result

def raw_cov(seq_x, seq_y):
    """Calculate covariance of two sequences

    Also return the mean of each sequence
    """
    x_bar = np.mean(seq_x)
    y_bar = np.mean(seq_y)

    # Covariance
    covar = np.dot(seq_x - x_bar, seq_y - y_bar)

    return covar, x_bar, y_bar

def raw_mom(seq):
    """Calculate the second moment of a sequence

    Also return the mean of the sequence
    """
    bar = np.mean(seq)
    mom = np.power(seq - bar, 2).sum()
    return mom, bar

def raw_emp_corr_inner(cov, mom1, mom2):
    """Calculate the correlation

    Given: covariance, moment 1, moment 2
    """
    den = np.sqrt(mom1 * mom2)
    return cov / den

# Emperical correlation
def raw_emp_corr(seq_x, seq_y):
    """Calculate the correlation of two sequences

    Also return the mean of both
    """
    # Covariance
    covar, x_bar, y_bar = raw_cov(seq_x, seq_y)

    # Standard deviations
    sig_x, _ = raw_mom(seq_x)
    sig_y, _ = raw_mom(seq_y)

    # Correlation
    return raw_emp_corr_inner(covar, sig_x, sig_y), x_bar, y_bar

def new_cov(C2_L1, mu_u1, mu_v1, n1, x2_u, x2_v):
    """Update running covariance with new samples

    Return the updated (running) value of correlation, mean of each sequence,
    and total sample count.

    Corresponds to eqn. III.6 in Bennett et. al.
    """

    # Get the count of the new set
    n2 = len(x2_u)
    assert n2 == len(x2_v)

    # Update the total count
    n = n1 + n2

    # Calculate the covariance and mean for just the new set
    C2_L2, mu_u2, mu_v2 = raw_cov(x2_u, x2_v)

    # Calculate the covariance itself according to III.6
    delt_u21 = mu_u2 - mu_u1
    delt_v21 = mu_v2 - mu_v1
    C2_L = C2_L1 + C2_L2 + ((n1 * n2)/n) * delt_u21 * delt_v21

    # Update the running mean according to II.3
    mu_u = mu_u1 + n2 * delt_u21 / n
    mu_v = mu_v1 + n2 * delt_v21 / n

    return C2_L, mu_u, mu_v, n

def new_mom(M2_L1, mu_1, n1, x2):
    """Update second moment with new samples

    Return the updated (running) value of correlation

    Corresponds to eqn. II.4 in Bennett et. al.
    """
    # Get the count of the new set
    n2 = len(x2)

    # Update the total count
    n = n1 + n2

    # Calculate the moment and mean of just the new test set
    M2_L2, mu_2 = raw_mom(x2)

    # Calculate the moment itself according to II.4
    delt_21 = mu_2 - mu_1
    M2_L = M2_L1 + M2_L2 + ((n1*n2)/n) * np.power(delt_21, 2)

    return M2_L

class TestIncrCorrMethods(unittest.TestCase):

    @staticmethod
    def _generate_rand_seqs(dims):
        """Get a pair of equally-sized random np arrays"""
        return np.random.rand(*dims), np.random.rand(*dims)

# One dimensional test-cases to check soundness of calculation

    def test_1D_short_rand_single(self):
        """Short 1-dimensional sequence of random numbers

        Calculate result in a single batch
        """
        test_x, test_y = self._generate_rand_seqs((1,10))
        tx_tens = torch.Tensor(test_x)
        ty_tens = torch.Tensor(test_y)

        # Our calculation
        corrobj = IncrCorr((1,1), 1)
        corrobj.update(tx_tens, ty_tens)
        mycorr = float(corrobj.retrieve().squeeze())

        # Reference calculation
        m = np.concatenate((test_x, test_y))
        npcorr = np.corrcoef(m)[0][1]

        self.assertAlmostEqual(npcorr, mycorr, places=5)

    def test_1D_short_rand_double(self):
        """Short 1-dimensional sequence of random numbers

        Calculate result in two batches
        """
        seq_len = 20
        test_x, test_y = self._generate_rand_seqs((1,seq_len))
        tx_tens = torch.Tensor(test_x)
        ty_tens = torch.Tensor(test_y)

        # Our calculation
        bat_bord = seq_len - 5
        corrobj = IncrCorr((1,1), 1)
        corrobj.update(tx_tens[:,:bat_bord], ty_tens[:,:bat_bord])
        corrobj.update(tx_tens[:,bat_bord:], ty_tens[:,bat_bord:])
        mycorr = float(corrobj.retrieve().squeeze())

        # Reference calculation
        m = np.concatenate((test_x, test_y))
        npcorr = np.corrcoef(m)[0][1]

        self.assertAlmostEqual(npcorr, mycorr, places=5)

    def test_1D_short_rand_zeroes(self):
        """Short 1-dimensional sequence of mostly zeroes

        Calculate result in a single batch
        """
        seq_len = 20
        test_x, test_y = np.zeros((1,seq_len)), np.zeros((1,seq_len))
        test_x[0,0] = 1
        test_y[0,0] = 2
        tx_tens = torch.Tensor(test_x)
        ty_tens = torch.Tensor(test_y)

        # Our calculation
        corrobj = IncrCorr((1,1), 1)
        corrobj.update(tx_tens, ty_tens)
        mycorr = float(corrobj.retrieve().squeeze())

        # Reference calculation
        m = np.concatenate((test_x, test_y))
        npcorr = np.corrcoef(m)[0][1]

        self.assertAlmostEqual(npcorr, mycorr, places=5)

# Multi-dimensional tests

    def test_2D_short_rand_single(self):
        """Short 2-dimensional sequence of random numbers

        Calculate result in a single batch
        """
        seq_len = 20
        test_x, test_y = self._generate_rand_seqs((2,seq_len))
        tx_tens = torch.Tensor(test_x)
        ty_tens = torch.Tensor(test_y)

        # Our calculation
        corrobj = IncrCorr((2,1), 1)
        corrobj.update(tx_tens, ty_tens)
        mycorr = corrobj.retrieve()

        # Reference calculation
        m0 = np.concatenate((np.expand_dims(test_x[0], 0), (np.expand_dims(test_y[0], 0))))
        npcorr0 = np.corrcoef(m0)[0][1]
        m1 = np.concatenate((np.expand_dims(test_x[1], 0), (np.expand_dims(test_y[1], 0))))
        npcorr1 = np.corrcoef(m1)[0][1]

        self.assertAlmostEqual(float(mycorr[0][0]), npcorr0, places=5)
        self.assertAlmostEqual(float(mycorr[1][0]), npcorr1, places=5)


    def test_3D_short_rand_double(self):
        """Short 3-dimensional sequence of random numbers

        Calculate result in two batches
        """
        seq_len = 20
        test_x, test_y = self._generate_rand_seqs((2,2,seq_len))
        tx_tens = torch.Tensor(test_x)
        ty_tens = torch.Tensor(test_y)

        # Our calculation
        bat_bord = seq_len - 5
        corrobj = IncrCorr((2,2,1), 2)
        corrobj.update(tx_tens[:,:,:bat_bord], ty_tens[:,:,:bat_bord])
        corrobj.update(tx_tens[:,:,bat_bord:], ty_tens[:,:,bat_bord:])
        mycorr = corrobj.retrieve()

        # Reference calculation
        m0 = np.concatenate((np.expand_dims(test_x[0,0], 0), (np.expand_dims(test_y[0,0], 0))))
        npcorr0 = np.corrcoef(m0)[0][1]
        m1 = np.concatenate((np.expand_dims(test_x[0,1], 0), (np.expand_dims(test_y[0,1], 0))))
        npcorr1 = np.corrcoef(m1)[0][1]
        m2 = np.concatenate((np.expand_dims(test_x[1,0], 0), (np.expand_dims(test_y[1,0], 0))))
        npcorr2 = np.corrcoef(m2)[0][1]
        m3 = np.concatenate((np.expand_dims(test_x[1,1], 0), (np.expand_dims(test_y[1,1], 0))))
        npcorr3 = np.corrcoef(m3)[0][1]

        self.assertAlmostEqual(float(mycorr[0][0]), npcorr0, places=5)
        self.assertAlmostEqual(float(mycorr[0][1]), npcorr1, places=5)
        self.assertAlmostEqual(float(mycorr[1][0]), npcorr2, places=5)
        self.assertAlmostEqual(float(mycorr[1][1]), npcorr3, places=5)

# Very long sequence

    def test_1D_long_rand_mult(self):
        """Long 1-dimensional sequence of random numbers

        Calculate result in many batches
        """
        seq_len = 100000
        bat_size = 10
        test_x, test_y = self._generate_rand_seqs((1,seq_len))
        tx_tens = torch.Tensor(test_x)
        ty_tens = torch.Tensor(test_y)

        # Our calculation
        corrobj = IncrCorr((1,1), 1)
        for idx in range(0, seq_len, bat_size):
            corrobj.update(tx_tens[:,idx:idx+bat_size], ty_tens[:,idx:idx+bat_size])
        mycorr = float(corrobj.retrieve().squeeze())

        # Reference calculation
        m = np.concatenate((test_x, test_y))
        npcorr = np.corrcoef(m)[0][1]

        self.assertAlmostEqual(npcorr, mycorr, places=5)

# Example use case

    def test_ex_use_case(self):
        # Comparison

        d1 = 46
        d2 = 4
        embed_dim = 512

        rst01inner = torch.rand(d1,d2,embed_dim)
        rst11inner = torch.rand(d1,d2,embed_dim)

        d1, d2, embed_dim = rst01inner.shape

        components_x = [list() for _ in range(embed_dim)]
        components_y = [list() for _ in range(embed_dim)]
        res_corr = []

        for x in range(d1):
            for y in range(d2):
                for comp in range(embed_dim):
                    components_x[comp].append(float(rst01inner[x,y,comp]))
                    components_y[comp].append(float(rst11inner[x,y,comp]))

        for comp in range(embed_dim):
            m = np.concatenate((
                np.expand_dims(components_x[comp], 0),
                np.expand_dims(components_y[comp], 0)))
            outmat = np.corrcoef(m)
            npcorr = outmat[0][1]

            res_corr.append(npcorr)

        np_total_corr = sum(res_corr) / embed_dim

        # Similar to calculation in code

        run_corr = IncrCorr((1,embed_dim), 0)
        tx_tens = torch.reshape(rst01inner, (-1, embed_dim))
        ty_tens = torch.reshape(rst11inner, (-1, embed_dim))
        run_corr.update(tx_tens, ty_tens)
        cur_corr = run_corr.retrieve()
        total_incr_corr = float(cur_corr.sum() / embed_dim)

        self.assertAlmostEqual(total_incr_corr, np_total_corr, places=5)

if __name__ == '__main__':
    unittest.main()


