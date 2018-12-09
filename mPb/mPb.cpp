#include "mPb.h"

#include "texton.h"
#include "textontools.h"

MultiscalePb::MultiscalePb(cv::Mat img, const std::string& textonsPath): 
	m_OrigImage(img), m_TextonPath(textonsPath) {
	calculateComponentImages();
}

bool MultiscalePb::computeGradients() {

	return true;
}

void MultiscalePb::calculateComponentImages() {
	cv::Mat imgLab;
	cv::cvtColor(m_OrigImage, imgLab, CV_BGR2Lab);
	std::vector<cv::Mat> imgLabComp;
	cv::split(imgLab, imgLabComp);
	cv::Mat lImage = imgLabComp[0];
	cv::Mat aImage = imgLabComp[1];
	cv::Mat bImage = imgLabComp[2];

	std::vector<Texton> textons;
	if (!TextonTools::readFromTextonsFile(m_TextonPath, textons)) {
		printf("Could not read the textons \n");
		exit(1);
	}

	cv::Mat textonImage;
	if (!TextonTools::convertToTextonImage(m_OrigImage, textons, textonImage)) {
		printf("Error when converting to texton image\n");
		exit(1);
	}

	m_ComponentImages["a"] = aImage;
	m_ComponentImages["b"] = bImage;
	m_ComponentImages["l"] = lImage;
	m_ComponentImages["t"] = textonImage;
}