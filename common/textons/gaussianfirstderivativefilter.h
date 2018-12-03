#ifndef GAUSSIANDERIVATIVEFILTER_H
#define GAUSSIANDERIVATIVEFILTER_H
#include "opencv2/opencv.hpp"
#include "textonkernel.h"

class GaussianFirstDerivativeFilter : public TextonKernel
{
public:
    GaussianFirstDerivativeFilter(int ksize, double orientation, double sigma, double sigmaXFactor = 1, double sigmaYFactor = 3);
    inline cv::Mat getKernel() { return m_Kernel; }
    void printValues();

private:
    double xComp(double x, double y) override;
    double yComp(double x, double y) override;
};

#endif // GAUSSIANDERIVATIVEFILTER_H
