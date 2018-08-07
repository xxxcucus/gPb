#include "textonkernel.h"

#define PI 3.14159265

TextonKernel::TextonKernel(int ksize, double orientation, double sigma, double sigmaXFactor, double sigmaYFactor):
    m_KernelSize(ksize), m_SigmaYFactor(sigmaYFactor), m_SigmaXFactor(sigmaXFactor), m_Sigma(sigma), m_Orientation(orientation) {
    m_Kernel = cv::Mat(cv::Size(m_KernelSize, m_KernelSize), CV_64FC1);

}

void TextonKernel::init() {
    int radius = m_KernelSize / 2;
    m_ScaleFactor = 0.0;

    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            double angle_rad = m_Orientation / 180 * PI;
            double i1 = i * cos(angle_rad) - j * sin(angle_rad);
            double j1 = i * sin(angle_rad) + j * cos(angle_rad);
            double val = xComp(i1, j1) + yComp(i1, j1);
            m_Kernel.at<double>(i + radius, j + radius) = val;
            m_ScaleFactor += std::abs(val);
        }
    }
    m_Kernel = m_Kernel / m_ScaleFactor;

    printf("Scale factor %f\n", m_ScaleFactor);
    printValues();
}

void TextonKernel::printValues() {
    int radius = m_KernelSize / 2;
    for (int i = -radius; i <= radius; ++i) {
        printf("[");
        for (int j = -radius; j <= radius; ++j) {
            printf("%f ",m_Kernel.at<double>(i + radius, j +  radius));
        }
        printf("]\n");
    }
}
