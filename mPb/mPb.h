#ifndef __MULTISCALE_PB_
#define __MULTISCALE_PB_

#include "opencv2/opencv.hpp"
#include <map>
#include <vector>

class MultiscalePb {
public:
	/**
	* img must be RGB image
	*/
	MultiscalePb(cv::Mat img, const std::string& textonPath, const std::map<std::string, std::vector<int>>& scales);

	/**
	* Computes the gradient images for all scales, orientations, and components
	*/
	bool computeGradients();

	/**
	* Calculate the multiscale edges
	*/
	void computeEdges();

	cv::Mat getEdges() {
		return m_GradImage;
	}

private:
	/**
	* Initialize alphas with equal values.
	*/
	void initializeAlphas();
	/**
	* Calculate L, a, b components from Lab conversion of image
	* and texton image corresponding to the quantized textons
	*/
	void calculateComponentImages();
	/**
	* Calculate gradient images for component with name compName
	* at the scale with index sIndex in m_Scales[compName].
	* One gradient image per orientation will be computed.
	*/
	std::vector<cv::Mat> calculateGradientImage(const std::string& compName, int sIndex);
	/**
	* Calculate gradient images for component with name compName.
	*/
	void calculateGradientImage(const std::string& compName);

	cv::Mat nonMaximumSuppression(const std::vector<cv::Mat> orientImgs, cv::Mat maxImage);

private:

	cv::Mat m_OrigImage;
	cv::Mat m_GradImage;

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