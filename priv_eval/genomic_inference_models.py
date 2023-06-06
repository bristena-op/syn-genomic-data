#compare the ldaf and recombination inference to the linear regression on real dat
import math
import random
import numpy as np
import pandas as pd


class BaseModel(object):
    PREC = 10 ** -4

    def __init__(self, SNPRefList):
        self.SNPRefList = SNPRefList

    def predict(self, seq, pos_index):

        probs = self.condProb(seq[:pos_index +1])
        r = random.random()
        if r < probs[0]:
            pred_value = 0
        elif r < probs[0] + probs[1]:
            pred_value = 1
        elif r < 1:
            pred_value = 2
        else:
            raise ValueError("Invalid random float number: %s" % (r))
        return pred_value

    def condProb(self, prefixSeq):
        return [0, 0, 0]


class PubLDModel(BaseModel):
    def __init__(self, SNPRefList, AF, LD):
        super(PubLDModel, self).__init__(SNPRefList)
        self.AF = AF
        self.LD = LD

    def condProb(self, prefixSeq):
        prefixLen = len(prefixSeq)
        assert prefixLen >= 1
        SNVLDTuple = self.LD[prefixLen - 1][0:2]
        probs = np.ones((3)) / 3
        # When there is no LD for the current SNV
        if len(SNVLDTuple) == 0:
            probs[0] = self.AF[prefixLen - 1][0] ** 2
            probs[1] = 2 * self.AF[prefixLen - 1][0] * self.AF[prefixLen - 1][1]
            probs[2] = 1 - probs[0] - probs[1]
        else:
            jointMat = self.pairwiseJoint(prefixLen - 1, SNVLDTuple[0], SNVLDTuple[1])
            probs = jointMat[prefixSeq[SNVLDTuple[0]], :]
            probs = probs / sum(probs)
        probs[np.nonzero(probs < self.PREC)] = self.PREC
        probs = probs / sum(probs)
        return probs

    '''
    Calculate the joint probability of two SNPs, namely, Pr[SNV1, SNV2]
    In our case, varSNP is the index of the current SNV, condSNP is the index of a previous SNV.
    D is the D_value between these two SNVs.
    '''

    def pairwiseJoint(self, SNV1, SNV2, D):
        #     D = LD[condSNP][varSNP]
        p_AB = self.AF[SNV2][0] * self.AF[SNV1][0] + D
        p_Ab = self.AF[SNV2][0] * self.AF[SNV1][1] - D
        p_aB = self.AF[SNV2][1] * self.AF[SNV1][0] - D
        p_ab = self.AF[SNV2][1] * self.AF[SNV1][1] + D
        jointMat = np.zeros((3, 3))
        jointMat[0, 0] = p_AB * p_AB
        jointMat[0, 1] = 2 * p_AB * p_Ab
        jointMat[0, 2] = p_Ab * p_Ab
        jointMat[1, 0] = 2 * p_AB * p_aB
        jointMat[1, 1] = 2 * p_AB * p_ab + 2 * p_Ab * p_aB
        jointMat[1, 2] = 2 * p_Ab * p_ab
        jointMat[2, 0] = p_aB * p_aB
        jointMat[2, 1] = 2 * p_aB * p_ab
        jointMat[2, 2] = p_ab * p_ab
        jointMat[np.nonzero(jointMat < 0)] = 1e-20
        #         jointMat[np.nonzero(jointMat < self.PREC)] = self.PREC
        jointMat = jointMat / sum(sum(jointMat))
        return jointMat


class DirectCondProbModel(BaseModel):
    def __init__(self, SNPRefList, directCondProbs, order):
        super(DirectCondProbModel, self).__init__(SNPRefList)
        self.directCondProbs = directCondProbs
        self.order = order

    def condProb(self, prefixSeq):
        prefixLen = len(prefixSeq)
        assert prefixLen >= 1
        idx = 0
        for i in range(min(self.order, prefixLen - 1), 0, -1):
            idx += prefixSeq[prefixLen - i - 1]
            idx = idx * 3
        probs = np.array(self.directCondProbs[len(prefixSeq) - 1][idx:(idx + 3)])
        probs[np.nonzero(probs < self.PREC)] = self.PREC
        probs = probs / sum(probs)
        return probs


class RecombModel(BaseModel):
    Ne = 11418  # Effective population size

    def __init__(self, SNPRefList, haplotype, geneticDist):
        super(RecombModel, self).__init__(SNPRefList)
        self.haplotype = np.array(haplotype)
        import pdb; pdb.set_trace()
        self.N = np.shape(self.haplotype)[1]
        self.geneticDist = geneticDist
        self.theta = self.computeTheta(self.N)
        self.mutateMatrix = self.computeMutateMatrix(self.theta, self.N)
        self.alpha = np.zeros((3, self.N, self.N))
        # Bias correction coefficients
        # From condlike.hpp in "http://stephenslab.uchicago.edu/software.html#hotspotter"
        self.a_ = -3.817e-01 + 6.350e-03 * len(SNPRefList) - 3.833e-05 * len(SNPRefList) * len(SNPRefList);
        self.b_ = -1.133e-01 - 2.600e-03 * len(SNPRefList) + 1.333e-05 * len(SNPRefList) * len(SNPRefList);

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

    def condProb(self, prefixSeq):
        prefixLen = len(prefixSeq)
        prefixProb = 1  # Note that this prefixProb is the probability of the prefix, not including the current snp (unlike prefixLen)
        if prefixLen == 1:
            m1 = np.repeat(np.array([self.haplotype[0, :]]), self.N, 0)
            m2 = np.repeat(np.array([self.haplotype[0, :]]).transpose(), self.N, 1)
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


