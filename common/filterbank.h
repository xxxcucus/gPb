#ifndef __FILTER_BANK__
#define __FILTER_BANK__

#include "opencv2/opencv.hpp"
#include <vector>
#include "textonkernel.h"

class FilterBank {
public:
	FilterBank() {
		create();
	}
	~FilterBank();
	std::vector<cv::Mat> runOnGrayScaleImage(const cv::Mat& img);

private:
	void create();

private:
	//filters used to compute the textons
	std::vector<TextonKernel*> m_FilterBank;
};

#endif