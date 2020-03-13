import numpy as np
import phase as phase

class BasisGenerator():

    def __init__(self, phaseGenerator, numBasis = 10):

        self.numBasis = numBasis
        self.phaseGenerator = phaseGenerator

    def basis(self):
        return None

    def basis_derivative(self):
        return None

    def basisMultiDoF(self, time, numDoF):
        basisSingleDoF = self.basis(time)

        basisMultiDoF = np.zeros((basisSingleDoF.shape[0] * numDoF, basisSingleDoF.shape[1] * numDoF))

        for i in range(numDoF):
            rowIndices = slice(i * basisSingleDoF.shape[0], (i + 1) * basisSingleDoF.shape[0])
            columnIndices = slice(i * basisSingleDoF.shape[1], (i + 1) * basisSingleDoF.shape[1])
            basisMultiDoF[rowIndices, columnIndices] = basisSingleDoF
        return basisMultiDoF

    def basisMultiDoFDerivative(self, time, numDoF):
        basisSingleDoFDeriv = self.basis_derivative(time)
        basisMultiDoFDerivative = np.zeros((basisSingleDoFDeriv.shape[0] * numDoF, basisSingleDoFDeriv.shape[1] * numDoF))

        for i in range(numDoF):
            rowIndices = slice(i * basisSingleDoFDeriv.shape[0], (i + 1) * basisSingleDoFDeriv.shape[0])
            columnIndices = slice(i * basisSingleDoFDeriv.shape[1], (i + 1) * basisSingleDoFDeriv.shape[1])
            basisMultiDoFDerivative[rowIndices, columnIndices] = basisSingleDoFDeriv
        return basisMultiDoFDerivative

    def total_basis(self, time, numDoF):
        basisMultiDoF = self.basisMultiDoF(time, numDoF)
        basisMultiDoFDerivative = self.basisMultiDoFDerivative(time, numDoF)
        return np.vstack((basisMultiDoF, basisMultiDoFDerivative))


class DMPBasisGenerator(BasisGenerator):

    def __init__(self, phaseGenerator, numBasis = 10, duration = 1, basisBandWidthFactor = 3):
        BasisGenerator.__init__(self, phaseGenerator, numBasis)

        self.basisBandWidthFactor = basisBandWidthFactor

        timePoints = np.linspace(0, duration, self.numBasis)
        self.centers = self.phaseGenerator.phase(timePoints)

        tmpBandWidth = np.hstack((self.centers[1:]-self.centers[0:-1], self.centers[-1] - self.centers[- 2]))

        # The Centers should not overlap too much (makes w almost random due to aliasing effect).Empirically chosen
        self.bandWidth = self.basisBandWidthFactor / (tmpBandWidth ** 2)


    def basis(self, time):

        phase = self.phaseGenerator.phase(time)

        diffSqr = np.array([((x - self.centers) ** 2) * self.bandWidth for x in phase])
        basis = np.exp(- diffSqr /  2)

        sumB = np.sum(basis, axis=1)
        basis = [column * phase / sumB for column in basis.transpose()]
        return np.array(basis).transpose()


class NormalizedRBFBasisGenerator(BasisGenerator):

    def __init__(self, phaseGenerator, numBasis = 10, duration = 1, basisBandWidthFactor = 3, numBasisOutside = 0):
        BasisGenerator.__init__(self, phaseGenerator, numBasis)

        self.basisBandWidthFactor = basisBandWidthFactor
        self.numBasisOutside = numBasisOutside
        basisDist = duration / (self.numBasis - 2 * self.numBasisOutside - 1)

        timePoints = np.linspace(-self.numBasisOutside * basisDist, duration + self.numBasisOutside * basisDist, self.numBasis)
        self.centers = self.phaseGenerator.phase(timePoints)

        tmpBandWidth = np.hstack((self.centers[1:]-self.centers[0:-1], self.centers[-1] - self.centers[- 2]))

        # The Centers should not overlap too much (makes w almost random due to aliasing effect).Empirically chosen
        self.bandWidth = self.basisBandWidthFactor / (tmpBandWidth ** 2)

    def basis(self, time):

        if isinstance(time, (float, int)):
            time = np.array([time])

        phase = self.phaseGenerator.phase(time)

        diffSqr = np.array([((x - self.centers) ** 2) * self.bandWidth for x in phase])
        basis = np.exp(- diffSqr / 2)

        sumB = np.sum(basis, axis=1)
        basis = [column / sumB for column in basis.transpose()]
        return np.array(basis).transpose()

    def basis_derivative(self, time):

        if isinstance(time, (float, int)):
            time = np.array([time])

        phase = self.phaseGenerator.phase(time)

        diffSqr = np.array([((x - self.centers) ** 2) * self.bandWidth for x in phase])
        basis = np.exp(- diffSqr / 2)

        multi_fac = np.array([(x - self.centers) * -self.bandWidth for x in phase])
        basis_derivative = np.multiply(multi_fac, basis)

        sumBD = np.sum(basis_derivative, axis=1)
        basis_derivative = [column / sumBD for column in basis_derivative.transpose()]
        return np.array(basis_derivative).transpose()



class NormalizedRhythmicBasisGenerator(BasisGenerator):

    def __init__(self, phaseGenerator,  numBasis = 10, duration = 1, basisBandWidthFactor = 3):

        BasisGenerator.__init__(self, phaseGenerator, numBasis)

        self.numBandWidthFactor = basisBandWidthFactor
        self.centers = np.linspace(0, 1, self.numBasis)

        tmpBandWidth = np.hstack((self.centers[1:]-self.centers[0:-1], self.centers[-1] - self.centers[- 2]))

        # The Centers should not overlap too much (makes w almost random due to aliasing effect).Empirically chosen
        self.bandWidth = self.basisBandWidthFactor / (tmpBandWidth ** 2)

    def basis(self):

        phase = self.getInputTensorIndex(0)

        diff = np.arraay([np.cos((phase - self.centers) * self.bandWidth * 2 * np.pi)])
        basis = np.exp(diff)

        sumB = np.sum(basis, axis=1)
        basis = [column / sumB for column in basis.transpose()]
        return np.array(basis).transpose()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    phaseGenerator = phase.LinearPhaseGenerator()
    basisGenerator = NormalizedRBFBasisGenerator(phaseGenerator, numBasis = 10, duration = 1, basisBandWidthFactor = 3, numBasisOutside = 1)

    time = np.linspace(0, 1, 100)
    basis = basisGenerator.basis(time)
    basisMultiDoF = basisGenerator.basisMultiDoF(time, 3)

    plt.figure()
    plt.plot(time, basis)

    plt.show()

    print('done')

