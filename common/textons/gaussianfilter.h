#ifndef GAUSSIANFILTER_H
#define GAUSSIANFILTER_H

#include "textonkernel.h"

class GaussianFilter : public TextonKernel
{
public:
    GaussianFilter(int ksize, double sigma);
    inline cv::Mat getKernel() { return m_Kernel; }
    void printValues();

private:
    double xComp(double x, double y) override;
    double yComp(double x, double y) override;
};

#endif // GAUSSIANFILTER_H
