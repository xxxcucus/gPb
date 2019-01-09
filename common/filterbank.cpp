#include "filterbank.h"

#include "gaussianfilter.h"
#include "gaussianfirstderivativefilter.h"
#include "gaussiansecondderivativefilter.h"
#include "texton.h"

FilterBank::~FilterBank() {
	for (auto f : m_FilterBank)
		delete f;
	m_FilterBank.clear();
}

void FilterBank::create() {
	double sigma = 1.0;

	for (int k = 0; k < 2; k++) {
		for (int i = 0; i < 4; ++i) {
			TextonKernel* tk = new GaussianFirstDerivativeFilter(5, double(i) * 22.5, (k + 1) * sigma);
			tk->init();
			m_FilterBank.push_back(tk);
		}
	}

	for (int k = 0; k < 2; k++) {
		for (int i = 0; i < 4; ++i) {
			TextonKernel* tk = new GaussianSecondDerivativeFilter(5 + (k + 1) * 4, double(i) * 22.5, (k + 1) * sigma);
			tk->init();
			m_FilterBank.push_back(tk);
		}
	}

	TextonKernel* tk = new GaussianFilter(9, sigma);
	tk->init();
	m_FilterBank.push_back(tk);

	/*for (unsigned int i = 0; i < m_FilterBank.size(); ++i) {
		printf("Matrix %u\n", i);
		m_FilterBank[i]->printValues();
		printf("\n");
	}*/
}

std::vector<cv::Mat> FilterBank::runOnGrayScaleImage(const cv::Mat &greyImg) {
	std::vector<cv::Mat> retVal;
	for (auto f : m_FilterBank) {
		cv::Mat filtImg;
		cv::filter2D(greyImg, filtImg, -1, f->getKernel(), cv::Point(-1, -1), CV_16S);
		cv::Mat rescaled;
		//TODO: is this needed?
		cv::convertScaleAbs(filtImg, rescaled, 5.0);
		retVal.push_back(filtImg);
	}
	return retVal;
}