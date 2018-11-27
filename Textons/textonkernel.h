#ifndef TEXTONKERNEL_H
#define TEXTONKERNEL_H

#include "opencv2/opencv.hpp"

/**
 * @brief The TextonKernel class models a kernel based on a gaussian
 */
class TextonKernel
{
public:
    TextonKernel(int ksize, double orientation, double sigma, double sigmaXFactor = 1, double sigmaYFactor = 3);
    inline cv::Mat getKernel() { return m_Kernel; }
    void printValues();
    void init();

private:
    virtual double xComp(double x, double y) = 0;
    virtual double yComp(double x, double y) = 0;

protected:

    int m_KernelSize = 5;   //must be odd for symmetry
    double m_SigmaYFactor = 3.0;
    double m_SigmaXFactor = 1.0;
    double m_Sigma = 1.0;
    cv::Mat m_Kernel;
    double m_ScaleFactor = 1.0;
    double m_Orientation; //in degrees
};

#endif // TEXTONKERNEL_H
