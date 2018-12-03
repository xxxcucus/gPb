#include "gaussiansecondderivativefilter.h"


GaussianSecondDerivativeFilter::GaussianSecondDerivativeFilter(int ksize, double orientation, double sigma, double sigmaXFactor, double sigmaYFactor) :
    TextonKernel(ksize, orientation, sigma, sigmaXFactor, sigmaYFactor) {
}

double GaussianSecondDerivativeFilter::xComp(double x, double y) {
    double expVal = exp(-(x * x / m_SigmaXFactor / m_SigmaXFactor + y * y / m_SigmaYFactor / m_SigmaYFactor ) / 2.0 / m_Sigma / m_Sigma);
    double factor = -( x * x / m_SigmaXFactor / m_SigmaXFactor / m_Sigma / m_Sigma - 1);
    return factor * expVal / m_SigmaXFactor / m_SigmaXFactor / m_Sigma / m_Sigma;

}

double GaussianSecondDerivativeFilter::yComp(double x, double y) {
    double expVal = exp(-(x * x / m_SigmaXFactor / m_SigmaXFactor + y * y / m_SigmaYFactor / m_SigmaYFactor ) / 2.0 / m_Sigma / m_Sigma);
    double factor = -( y * y / m_SigmaYFactor / m_SigmaYFactor / m_Sigma / m_Sigma - 1);
    return factor * expVal / m_SigmaYFactor / m_SigmaYFactor / m_Sigma / m_Sigma;
}

