#include "gaussianfirstderivativefilter.h"
#include <cmath>

GaussianFirstDerivativeFilter::GaussianFirstDerivativeFilter(int ksize, double orientation, double sigma, double sigmaXFactor, double sigmaYFactor) :
    TextonKernel(ksize, orientation, sigma, sigmaXFactor, sigmaYFactor) {
}

double GaussianFirstDerivativeFilter::xComp(double x, double y) {
    double expVal = exp(-(x * x / m_SigmaXFactor / m_SigmaXFactor + y * y / m_SigmaYFactor / m_SigmaYFactor ) / 2.0 / m_Sigma / m_Sigma);
    return -x / m_SigmaXFactor / m_SigmaXFactor * expVal;
}

double GaussianFirstDerivativeFilter::yComp(double x, double y) {
    double expVal = exp(-(x * x / m_SigmaXFactor / m_SigmaXFactor + y * y / m_SigmaYFactor / m_SigmaYFactor ) / 2.0 / m_Sigma / m_Sigma);
    return -y / m_SigmaYFactor / m_SigmaYFactor * expVal;
}
