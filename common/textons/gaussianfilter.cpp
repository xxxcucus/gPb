#include "gaussianfilter.h"

GaussianFilter::GaussianFilter(int ksize, double sigma) :
    TextonKernel(ksize, 45.0, sigma, 1.0, 1.0) {
}

double GaussianFilter::xComp(double x, double y) {
    double expVal = exp(-(x * x / m_SigmaXFactor / m_SigmaXFactor + y * y / m_SigmaYFactor / m_SigmaYFactor ) / 2.0 / m_Sigma / m_Sigma);
    return expVal / 2;
}

double GaussianFilter::yComp(double x, double y) {
    return xComp(x, y);
}
