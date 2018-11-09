#include "discdirectmasks.h"
#include "discinversemasks.h"
#include "pbdetector.h"
#include <QDebug>
#include <QCoreApplication>
#include "opencv2/opencv.hpp"
#include "cudaimage.h"

int main(int argc, char* argv[])
{
    /*****************************
     * test disc direct masks
     * *********************************/

    DiscDirectMasks ddm(5);

    for (int i = 0; i < 8; ++i) {
        qDebug() << "Testing half disc " << i;

        std::vector<QPoint> points = ddm.getHalfDiscPoints(i);
        for (auto p : points) {
            qDebug() << p.x() << "-" << p.y();
        }
    }

    //return 0;

    /*************************************
     * test disc inverse masks
     * ***********************************/

    qDebug() << "Testing inverse disc";
    DiscInverseMasks dim(5);
    std::vector<std::vector<int>>& points = dim.getHalfDiscInfluencePoints();
    for (auto p : points) {
        QString ps;
        for (auto pp : p)
            ps += " " + QString::number(pp);
        qDebug() << ps;
    }

    //return 0;

	/******************************************
	* cudaMPb
	********************************************/

	std::string sourcePath1 = "D:/cucu/temp/gPb/mPb/Sternchen2016.jpg";
	cv::Mat img1 = cv::imread(sourcePath1);
	CudaImage cudaImg(img1.data, img1.cols, img1.rows, 10);

	return 0;

    /******************************************
     * test gradient calculation
     * ****************************************/

    std::string sourcePath = "D:/ProjectsOpenCV/gPb/mPb/Sternchen2016.jpg";
    std::string lCompPath = "D:/ProjectsOpenCV/gPb/mPb/LComp.png";
    std::string grad0Path = "D:/ProjectsOpenCV/gPb/mPb/grad0.png";
    std::string grad1Path = "D:/ProjectsOpenCV/gPb/mPb/grad1.png";
    std::string grad2Path = "D:/ProjectsOpenCV/gPb/mPb/grad2.png";
    std::string grad3Path = "D:/ProjectsOpenCV/gPb/mPb/grad3.png";
    cv::Mat img = cv::imread(sourcePath);
    int nRows = img.rows / 3;
    int nCols = img.cols / 3;
    cv::Mat resImg;
    cv::resize(img, resImg, cv::Size(nCols, nRows));
    cv::Mat imgLab;
    cv::cvtColor(resImg, imgLab, CV_BGR2Lab);
    std::vector<cv::Mat> imgLabComp;
    cv::split(imgLab, imgLabComp);
    cv::imwrite(lCompPath, imgLabComp[2]);
    qDebug() << "Saving L component" << lCompPath.c_str();

    PbDetector pbd(3, imgLabComp[2]);
    pbd.calculateGradients();
    qDebug() << "Calculate gradients";
    cv::imwrite(grad0Path, pbd.getGradientImage(0));
    cv::imwrite(grad1Path, pbd.getGradientImage(1));
    cv::imwrite(grad2Path, pbd.getGradientImage(2));
    cv::imwrite(grad3Path, pbd.getGradientImage(3));

    return 0;
}
