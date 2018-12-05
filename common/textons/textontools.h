#ifndef _TEXTON_TOOLS_
#define _TEXTON_TOOLS_

#include "texton.h"
#include "opencv2/opencv.hpp"

class TextonTools {
public:	
	static bool readFromTextonsFile(const std::string& path, std::vector<Texton>& textons);
	static bool convertToTextonImage(cv::Mat img, const std::vector<Texton> m_QuantTextons, cv::Mat& result);
	
	/**
	* @brief getTexton - gets a texton from a set of filtered images
	* @param filtImgs
	* @param x, y - position in the images the texton is calculated for
	*/
	static Texton getTexton(const std::vector<cv::Mat>& filtImgs, int x, int y);

	static int getClosestClusterCenter(const Texton& t, const std::vector<Texton>& cluster);
};


#endif