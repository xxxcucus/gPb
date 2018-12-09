#ifndef __MULTISCALE_PB_
#define __MULTISCALE_PB_

#include "opencv2/opencv.hpp"
#include <map>
#include <vector>

class MultiscalePb {

	/**
	* img must be RGB image
	*/
	MultiscalePb(cv::Mat img, const std::string& textonPath);

	void setScales(const std::string& comp, const std::vector<int>& scales) {
		m_Scales[comp] = scales;
	}

	/**
	* Computes the gradient images for all scales, orientations, and components
	*/
	bool computeGradients();

private:
	void calculateComponentImages();

private:

	cv::Mat m_OrigImage;

	/**
	* Path to quantized textons.
	*/
	std::string m_TextonPath;

	std::map<std::string, cv::Mat> m_ComponentImages;

	/**
	* For each component and each orientation a vector of gradient images, one for each scale
	*/
	std::map <std::string, std::vector<std::vector<cv::Mat>>> m_GradientImages;
	/**
	* For each component and each orientation a vector of weights, one for each scale
	*/
	std::map<std::string, std::vector<std::vector<double>>> m_Alphas;
	/**
	* For each component a vector of the scales used. 
	*/
	std::map<std::string, std::vector<int>> m_Scales;
	const std::vector<std::string> m_ComponentNames = { "a", "b", "l", "t" };
	const std::vector<double> m_Orientations = { 0, CV_PI / 4.0, CV_PI / 2.0, CV_PI * 3.0 / 4.0 };
};

















#endif