#include "discdirectmasks.h"
#include "discinversemasks.h"
#include "pbdetector.h"
#include <QDebug>
#include <QCoreApplication>
#include "opencv2/opencv.hpp"
#include "cudampb.h"
#include <chrono>

int main(int argc, char* argv[])
{
    /*****************************
     * test disc direct masks
     * *********************************/

    /*DiscDirectMasks ddm(5);

    for (int i = 0; i < 8; ++i) {
        qDebug() << "Testing half disc " << i;

        std::vector<QPoint> points = ddm.getHalfDiscPoints(i);
        for (auto p : points) {
            qDebug() << p.x() << "-" << p.y();
        }
    }*/

    //return 0;

    /*************************************
     * test disc inverse masks
     * ***********************************/

    /*qDebug() << "Testing inverse disc";
    DiscInverseMasks dim(5);
    std::vector<std::vector<int>>& points = dim.getHalfDiscInfluencePoints();
    for (auto p : points) {
        QString ps;
        for (auto pp : p)
            ps += " " + QString::number(pp);
        qDebug() << ps;
    }*/

    //return 0;

	/******************************************
	* cudaMPb
	********************************************/

	std::string sourcePath1 = "Sternchen2016.jpg";
	std::string grad00Path = "grad0.png";
	std::string grad01Path = "grad1.png";
	std::string grad02Path = "grad2.png";
	std::string grad03Path = "grad3.png";
	cv::Mat img1 = cv::imread(sourcePath1, CV_LOAD_IMAGE_GRAYSCALE);
	printf("BlaBla1\n");
	CudaMPb cudaImg(img1.data, img1.cols, img1.rows, 5);
	if (!cudaImg.wasSuccessfullyCreated()) {
		printf("Error in constructor %s. Exiting\n", cudaImg.getErrorString());
		exit(1);
	}
	printf("BlaBla2\n");
	auto start = std::chrono::high_resolution_clock::now();
	if (!cudaImg.execute()) {
		printf("Error when executing %s. Exiting\n", cudaImg.getErrorString());
		exit(1);
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	printf("GPU runtime %d\n", int(duration.count()));
	printf("BlaBla3\n");
	cv::Mat grad0(img1.rows, img1.cols, CV_64FC1, cudaImg.getGradientImage(0));
	cv::imwrite(grad00Path, grad0);
	cv::Mat grad1(img1.rows, img1.cols, CV_64FC1, cudaImg.getGradientImage(1));
	cv::imwrite(grad01Path, grad1);
	cv::Mat grad2(img1.rows, img1.cols, CV_64FC1, cudaImg.getGradientImage(2));
	cv::imwrite(grad02Path, grad2);
	cv::Mat grad3(img1.rows, img1.cols, CV_64FC1, cudaImg.getGradientImage(3));
	cv::imwrite(grad03Path, grad3);
	printf("BlaBla4\n");

	return 0;

    /******************************************
     * test gradient calculation with CPU
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
