#include <QDebug>
#include <QCoreApplication>
#include "opencv2/opencv.hpp"
#include <chrono>
#include <string>

#include "cudapbdetector.h"
#include "discdirectmasks.h"
#include "discinversemasks.h"
#include "pbdetector.h"
#include "texton.h"
#include "textontools.h"
#include "mPb.h"


void calculateGradients(cv::Mat inputImg, std::string imgName, int scale);

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

	/*std::string sourcePath = "Sternchen2016.jpg";
	std::string cuda_grad0Path = "cuda_grad0.png";
	std::string cuda_grad1Path = "cuda_grad1.png";
	std::string cuda_grad2Path = "cuda_grad2.png";
	std::string cuda_grad3Path = "cuda_grad3.png";

	std::string lCompPath = "LComp.png";
	std::string aCompPath = "AComp.png";
	std::string bCompPath = "BComp.png";
	std::string textonPath = "textonComp.png";

	std::string cpu_grad0Path = "cpu_grad0.png";
	std::string cpu_grad1Path = "cpu_grad1.png";
	std::string cpu_grad2Path = "cpu_grad2.png";
	std::string cpu_grad3Path = "cpu_grad3.png";*/

	/******************************************
	* resize, convert to Lab and extract L component
	* ****************************************/

	/*cv::Mat img = cv::imread(sourcePath);
	int nRows = img.rows / 3;
	int nCols = img.cols / 3;
	cv::Mat resImg;
	cv::resize(img, resImg, cv::Size(nCols, nRows));
	cv::Mat imgLab;
	cv::cvtColor(resImg, imgLab, CV_BGR2Lab);
	std::vector<cv::Mat> imgLabComp;
	cv::split(imgLab, imgLabComp);
	cv::Mat lImage = imgLabComp[0];
	cv::Mat aImage = imgLabComp[1];
	cv::Mat bImage = imgLabComp[2];
	cv::imwrite(lCompPath, lImage);
	cv::imwrite(aCompPath, aImage);
	cv::imwrite(bCompPath, bImage);
	//qDebug() << "Saving L component" << lCompPath.c_str();

	cv::Mat img_for_textons = cv::imread(sourcePath, cv::IMREAD_GRAYSCALE);
	cv::Mat res_img_textons;
	cv::resize(img_for_textons, res_img_textons, cv::Size(nCols, nRows));
	std::vector<Texton> textons;
	std::string textonQuantPath = "textons.txt";
	if (!TextonTools::readFromTextonsFile(textonQuantPath, textons)) {
		printf("Could not read the textons \n");
		exit(1);
	}

	cv::Mat textonImage;
	if (!TextonTools::convertToTextonImage(res_img_textons, textons, textonImage)) {
		printf("Error when converting to texton image\n");
		exit(1);
	}

	cv::imwrite(textonPath, textonImage);

	int scale = 5;*/

    /******************************************
     * test gradient calculation with CPU
     * ****************************************/
	/*if (false) {
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
	}*/

	/******************************************
	* cudaMPb
	********************************************/

	/*calculateGradients(lImage, "LIMAGE", scale);
	calculateGradients(aImage, "AIMAGE", scale);
	calculateGradients(bImage, "BIMAGE", scale);
	calculateGradients(textonImage, "TIMAGE", scale);

	calculateGradients(lImage, "LIMAGE", 2 * scale);
	calculateGradients(aImage, "AIMAGE", 2 * scale);
	calculateGradients(bImage, "BIMAGE", 2 * scale);
	calculateGradients(textonImage, "TIMAGE", 2 * scale);*/

	/******************************************
	* MPb detector
	********************************************/

	std::string sourcePath = "Sternchen2016.jpg";
	cv::Mat img = cv::imread(sourcePath);
	int nRows = img.rows / 3;
	int nCols = img.cols / 3;
	cv::Mat resImg = cv::Mat(nRows, nCols, img.type());
	cv::resize(img, resImg, cv::Size(nCols, nRows));
	printf("Image size %d - %d\n", nRows, nCols);

	std::map<std::string, std::vector<int>> mapScales;
	std::vector<int> scales = { 3, 5, 7 };
	mapScales["l"] = scales;
	mapScales["a"] = scales;
	mapScales["b"] = scales;
	mapScales["t"] = scales;

	MultiscalePb detector(resImg, "textons.txt", mapScales);
	auto grad_start = std::chrono::high_resolution_clock::now();
	detector.computeGradients();
	auto grad_stop = std::chrono::high_resolution_clock::now();
	auto grad_duration = std::chrono::duration_cast<std::chrono::milliseconds>(grad_stop - grad_start);
	printf("Multiscale gradients runtime(ms) %d\n", int(grad_duration.count()));
	detector.computeEdges();
	cv::Mat edges = detector.getEdges();
	cv::imwrite("edges.png", edges);
	
	return 0;
}


void calculateGradients(cv::Mat inputImg, std::string imgName, int scale) {
	CudaPbDetector cudaImg(inputImg.data, inputImg.cols, inputImg.rows, scale);
	if (!cudaImg.wasSuccessfullyCreated()) {
		printf("Error in constructor %s. Exiting\n", cudaImg.getErrorString());
		exit(1);
	}
	auto cuda_start = std::chrono::high_resolution_clock::now();
	if (!cudaImg.executeChunk()) {
		printf("Error when executing %s. Exiting\n", cudaImg.getErrorString());
		exit(1);
	}
	auto cuda_stop = std::chrono::high_resolution_clock::now();
	auto cuda_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cuda_stop - cuda_start);
	printf("GPU runtime(ms) %d\n", int(cuda_duration.count()));
	cv::Mat cuda_grad0(inputImg.rows, inputImg.cols, CV_64FC1, cudaImg.getGradientImage(0));
	cv::imwrite(imgName + "grad_0" + std::to_string(scale) + ".png", cuda_grad0);
	cv::Mat cuda_grad1(inputImg.rows, inputImg.cols, CV_64FC1, cudaImg.getGradientImage(1));
	cv::imwrite(imgName + "grad_1" + std::to_string(scale) + ".png", cuda_grad1);
	cv::Mat cuda_grad2(inputImg.rows, inputImg.cols, CV_64FC1, cudaImg.getGradientImage(2));
	cv::imwrite(imgName + "grad_2" + std::to_string(scale) + ".png", cuda_grad2);
	cv::Mat cuda_grad3(inputImg.rows, inputImg.cols, CV_64FC1, cudaImg.getGradientImage(3));
	cv::imwrite(imgName + "grad_3" + std::to_string(scale) + ".png", cuda_grad3);
}
