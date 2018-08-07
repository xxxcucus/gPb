#ifndef GAUSSIANSECONDDERIVATIVEFILTER_H
#define GAUSSIANSECONDDERIVATIVEFILTER_H
#include "opencv2/opencv.hpp"
#include "textonkernel.h"

class GaussianSecondDerivativeFilter : public TextonKernel
{
public:
    GaussianSecondDerivativeFilter(int ksize, double orientation, double sigma, double sigmaXFactor = 1, double sigmaYFactor = 3);
    inline cv::Mat getKernel() { return m_Kernel; }
    void printValues();

private:
    //second partial derivative in x
    double xComp(double x, double y) override;
    //second partial derivative in y
    double yComp(double x, double y) override;
};

#endif // GAUSSIANSECONDDERIVATIVEFILTER_H
