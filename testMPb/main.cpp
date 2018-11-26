#include <QDebug>
#include <QCoreApplication>
#include "opencv2/opencv.hpp"
#include <chrono>
#include <string>

#include "cudampb.h"
#include "discdirectmasks.h"
#include "discinversemasks.h"
#include "pbdetector.h"

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
	* setting paths
	* ****************************************/

	std::string sourcePath = "Sternchen2016.jpg";
	std::string cuda_grad0Path = "cuda_grad0.png";
	std::string cuda_grad1Path = "cuda_grad1.png";
	std::string cuda_grad2Path = "cuda_grad2.png";
	std::string cuda_grad3Path = "cuda_grad3.png";

	std::string lCompPath = "LComp.png";
	std::string cpu_grad0Path = "cpu_grad0.png";
	std::string cpu_grad1Path = "cpu_grad1.png";
	std::string cpu_grad2Path = "cpu_grad2.png";
	std::string cpu_grad3Path = "cpu_grad3.png";

	/******************************************
	* resize, convert to Lab and extract L component
	* ****************************************/

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
	//qDebug() << "Saving L component" << lCompPath.c_str();

	int scale = 5;

    /******************************************
     * test gradient calculation with CPU
     * ****************************************/

	printf("Calculate gradients with CPU\n");
	
    PbDetector pbd(scale, imgLabComp[2]);
	auto cpu_start = std::chrono::high_resolution_clock::now();
    pbd.calculateGradients();
	auto cpu_stop = std::chrono::high_resolution_clock::now();
	auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_stop - cpu_start);
	printf("CPU runtime(ms) %d\n", int(cpu_duration.count()));

    cv::imwrite(cpu_grad0Path, pbd.getGradientImage(0));
    cv::imwrite(cpu_grad1Path, pbd.getGradientImage(1));
    cv::imwrite(cpu_grad2Path, pbd.getGradientImage(2));
    cv::imwrite(cpu_grad3Path, pbd.getGradientImage(3));



	/******************************************
	* cudaMPb
	********************************************/

	CudaMPb cudaImg(imgLabComp[2].data, imgLabComp[2].cols, imgLabComp[2].rows, scale);
	if (!cudaImg.wasSuccessfullyCreated()) {
		printf("Error in constructor %s. Exiting\n", cudaImg.getErrorString());
		exit(1);
	}
	auto cuda_start = std::chrono::high_resolution_clock::now();
	if (!cudaImg.execute()) {
		printf("Error when executing %s. Exiting\n", cudaImg.getErrorString());
		exit(1);
	}
	auto cuda_stop = std::chrono::high_resolution_clock::now();
	auto cuda_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cuda_stop - cuda_start);
	printf("GPU runtime(ms) %d\n", int(cuda_duration.count()));
	cv::Mat cuda_grad0(imgLabComp[2].rows, imgLabComp[2].cols, CV_64FC1, cudaImg.getGradientImage(0));
	cv::imwrite(cuda_grad0Path, cuda_grad0);
	cv::Mat cuda_grad1(imgLabComp[2].rows, imgLabComp[2].cols, CV_64FC1, cudaImg.getGradientImage(1));
	cv::imwrite(cuda_grad1Path, cuda_grad1);
	cv::Mat cuda_grad2(imgLabComp[2].rows, imgLabComp[2].cols, CV_64FC1, cudaImg.getGradientImage(2));
	cv::imwrite(cuda_grad2Path, cuda_grad2);
	cv::Mat cuda_grad3(imgLabComp[2].rows, imgLabComp[2].cols, CV_64FC1, cudaImg.getGradientImage(3));
	cv::imwrite(cuda_grad3Path, cuda_grad3);
	return 0;
}
