import math
import random
import numpy as np
import pandas as pd


class RecombModel(object):
    PREC = 10 ** -4
    Ne = 11418  # Effective population size


    def __init__(self,  geneticDistFileName):

        geneticDist , SNPRefList= self.load_aux_data( geneticDistFileName)
        self.geneticDist = geneticDist

        # Bias correction coefficients
        # From condlike.hpp in "http://stephenslab.uchicago.edu/software.html#hotspotter"


        self.__name__ = "RecombModel"
        self.datatype = pd.DataFrame
    def load_aux_data(self, geneticDistFileName):
        dat = pd.read_csv(geneticDistFileName, sep='\t', header=None)

        dat = dat.to_numpy()
        geneticDist = dat[:, 1]
        SNPRefList = dat[:,0]
        return geneticDist, SNPRefList

    def fit(self, data):
        self.data = data
        SNPRefList = data.columns
        haplotype =  np.transpose(data.to_numpy())
        self.SNPRefList = SNPRefList
        self.a_ = -3.817e-01 + 6.350e-03 * len(SNPRefList) - 3.833e-05 * len(SNPRefList) * len(SNPRefList);
        self.b_ = -1.133e-01 - 2.600e-03 * len(SNPRefList) + 1.333e-05 * len(SNPRefList) * len(SNPRefList);

        self.haplotype = np.array(list(map(lambda u: u[:200], haplotype)))
        self.N = np.shape(self.haplotype)[1]
        self.theta = self.computeTheta(self.N)
        self.mutateMatrix = self.computeMutateMatrix(self.theta, self.N)
        self.alpha = np.zeros((3, self.N, self.N))

        n = len(self.SNPRefList)
        seq = [-1] * n
        for i in range(1, n + 1):
            probs = self.compute_prob(seq[:i], self.haplotype)
            r = random.random()
            if r < probs[0]:
                seq[i - 1] = 0
            elif r < probs[0] + probs[1]:
                seq[i - 1] = 1
            elif r < 1:
                seq[i - 1] = 2
            else:
                raise ValueError("Invalid random float number: %s" % (r))
        return seq

    def computeTheta(self, N):
        theta = 0
        for i in range(1, N):
            theta += 1.0 / i
        theta = 1.0 / theta
        theta *= 0.1
        return theta

    def computeMutateMatrix(self, theta, N):
        r = theta * 0.5 / (theta + N)
        return np.array([[(1 - r) * (1 - r), 2 * r * (1 - r), r * r],
                         [r * (1 - r), r * r + (1 - r) * (1 - r), r * (1 - r)],
                         [r * r, 2 * r * (1 - r), (1 - r) * (1 - r)]])

    def compute_prob(self, prefixSeq, data):
        # import pdb; pdb.set_trace()
        prefixLen = len(prefixSeq)
        prefixProb = 1  # Note that this prefixProb is the probability of the prefix, not including the current snp (unlike prefixLen)
        if prefixLen == 1:
            m1 = np.repeat(np.array([data[0, :]]), self.N, 0)
            m2 = np.repeat(np.array([data[0, :]]).transpose(), self.N, 1)
            rowIdx = np.array(m1 + m2, dtype=np.intp)
            for columnIdx in range(3):
                self.alpha[columnIdx, :, :] = self.mutateMatrix[rowIdx, columnIdx] * 1.0 / (self.N * self.N)
        else:
            # suppose x is a state tuple (x_1, x_2)
            # alpha_{j+1} (x) = r_{j+1} (x) * ( p*p*alpha_j(x) + p*q*sum_{y_1 = x_1}(alpha_j(y)) + p*q*sum_{y_2 = x_2}(alpha_j(y) + q*q*sum_{y}(alpha_j(y)))
            #             p = math.exp(-4*self.Ne*self.geneticDist[prefixLen-1] * 1.0 / self.N)
            if self.geneticDist[prefixLen - 1] == 0.0:
                self.geneticDist[prefixLen - 1] = 1e-8
            p = math.exp(-self.biasCorrection(4 * self.Ne * self.geneticDist[prefixLen - 1],
                                              int(self.SNPRefList[prefixLen - 1]) - int(
                                                  self.SNPRefList[prefixLen - 2])) * 1.0 / self.N)
            #             p = math.exp(-4*self.Ne*self.geneticDist[prefixLen-1] * 1.0 / self.N)
            q = (1 - p) / self.N
            alpha_j = self.alpha[prefixSeq[prefixLen - 2], :, :]
            # term 1
            term_1 = p * p * alpha_j
            # term 2 and term 3
            sum_vec1 = np.array([np.sum(alpha_j, axis=0)])
            sum_vec2 = np.array([np.sum(alpha_j, axis=1)])
            sum_mat1 = np.repeat(sum_vec1, self.N, 0)
            sum_mat2 = np.repeat(sum_vec2.transpose(), self.N, 1)
            term_2_3 = p * q * (sum_mat1 + sum_mat2)
            # term 4
            prefixProb = np.sum(sum_vec1)
            term_4 = q * q * prefixProb

            # alpha_{j+1}
            term = term_1 + term_2_3 + term_4
            m1 = np.repeat(np.array([self.haplotype[prefixLen - 1, :]]), self.N, 0)
            m2 = np.repeat(np.array([self.haplotype[prefixLen - 1, :]]).transpose(), self.N, 1)
            rowIdx = np.array(m1 + m2, dtype=np.intp)
            for columnIdx in range(3):
                self.alpha[columnIdx, :, :] = self.mutateMatrix[rowIdx, columnIdx] * term
            # Renormalize alpha, to avoid float number precision problem when the sequence gets longer and longer, and alpha gets smaller and smaller
            # Since we only need the conditional probabilities, but not the probability of the prefix sequence, it doesn't matter if we scale alpha by a constant.
            # Here we scale it by dividing prefixProb so that alpha sums to 1. But other scale methods also work, only if we maintain the relative ratio between different alpha elements
            self.alpha = self.alpha / prefixProb
        probs = np.sum(np.sum(self.alpha, axis=2), axis=1)
        # Try weighting this with a uniform distribution to avoid overfit
        weight = 1.0
        return weight * probs + (1 - weight) * (np.array([0.25, 0.5, 0.25]))

    def biasCorrection(self, rate, phyDist):
        rho = rate / phyDist
        return rate * 0.0015

    def to_haplo(self, seq):
        h0, h1 = [], []
        for item in seq:
            if item == 0:
                h0.append(0)
                h1.append(0)
            elif item == 2:
                h0.append(1)
                h1.append(1)
            elif item == 1:
                r = random.random()
                if r <= 0.5:
                    h0.append(1)
                    h1.append(0)
                else:
                    h0.append(0)
                    h1.append(1)
        return h0, h1

    def generate_samples(self, no_samples):
        samples = []
        for i in range(int(no_samples/2)):
            seq = self.fit(self.data)
            h0, h1 = self.to_haplo(seq)
            samples.append(h0)
            samples.append(h1)
        # samples = np.transpose(np.array(samples))
        return pd.DataFrame(samples, columns=self.SNPRefList)
