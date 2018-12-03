#ifndef _TEXTON_TOOLS_
#define _TEXTON_TOOLS_

#include "texton.h"
#include "opencv2/opencv.hpp"

class TextonTools {
public:	
	static bool readFromTextonsFile(const std::string& path, std::vector<Texton>& textons);
	static bool convertToTextonImage(cv::Mat img, cv::Mat result);
};


#endif