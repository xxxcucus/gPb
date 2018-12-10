#include "mPb.h"

#include "texton.h"
#include "textontools.h"
#include "cudapbdetector.h"

#include <chrono>

MultiscalePb::MultiscalePb(cv::Mat img, const std::string& textonsPath, const std::map<std::string, std::vector<int>>& scales):
	m_OrigImage(img), m_TextonPath(textonsPath), m_Scales(scales) {
	calculateComponentImages();
	initializeAlphas();
}

bool MultiscalePb::computeGradients() {
	for (auto comp : m_ComponentNames) {
		calculateGradientImage(comp);
	}
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

	cv::Mat grayscaleImage;
	cv::cvtColor(m_OrigImage, grayscaleImage, CV_BGR2GRAY);

	cv::Mat textonImage;
	if (!TextonTools::convertToTextonImage(grayscaleImage, textons, textonImage)) {
		printf("Error when converting to texton image\n");
		exit(1);
	}

	m_ComponentImages["a"] = aImage;
	m_ComponentImages["b"] = bImage;
	m_ComponentImages["l"] = lImage;
	m_ComponentImages["t"] = textonImage;
}


void MultiscalePb::calculateGradientImage(const std::string& compName) {
	std::vector<std::vector<cv::Mat>> images;

	for (auto orient : m_Orientations) {
		images.push_back(std::vector<cv::Mat>());
	}

	for (unsigned int i = 0; i < m_Scales[compName].size(); ++i) {
		std::vector<cv::Mat> orientImgs = calculateGradientImage(compName, i);
		for (unsigned int j = 0; j < orientImgs.size(); ++j) {
			images[j].push_back(orientImgs[j]);
		}
	}

	m_GradientImages[compName] = images;
}

std::vector<cv::Mat> MultiscalePb::calculateGradientImage(const std::string& compName, int sIndex) {
	//TODO: error checking
	int scale = m_Scales[compName][sIndex];
	cv::Mat inputImg = m_ComponentImages[compName];

	printf("Calculate gradient images for %s component\n", compName.c_str());
	CudaPbDetector cudaImg(inputImg.data, inputImg.cols, inputImg.rows, scale);
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

	std::vector<cv::Mat> retVal;

	cv::Mat cuda_grad0(inputImg.rows, inputImg.cols, CV_64FC1, cudaImg.getGradientImage(0));
	retVal.push_back(cuda_grad0.clone());
	cv::imwrite(compName + "_grad_0_" + std::to_string(scale) + ".png", cuda_grad0);
	cv::Mat cuda_grad1(inputImg.rows, inputImg.cols, CV_64FC1, cudaImg.getGradientImage(1));
	retVal.push_back(cuda_grad1.clone());
	cv::imwrite(compName + "_grad_1_" + std::to_string(scale) + ".png", cuda_grad1);
	cv::Mat cuda_grad2(inputImg.rows, inputImg.cols, CV_64FC1, cudaImg.getGradientImage(2));
	retVal.push_back(cuda_grad2.clone());
	cv::imwrite(compName + "_grad_2_" + std::to_string(scale) + ".png", cuda_grad2);
	cv::Mat cuda_grad3(inputImg.rows, inputImg.cols, CV_64FC1, cudaImg.getGradientImage(3));
	retVal.push_back(cuda_grad3.clone());
	cv::imwrite(compName + "_grad_3_" + std::to_string(scale) + ".png", cuda_grad3);

	return retVal;
}

//TODO: consistency check 
void MultiscalePb::initializeAlphas() {
	int count = 0;
	for (auto compName : m_ComponentNames) {
		count += int(m_Orientations.size() * m_Scales[compName].size());
	}

	if (!count)
		return;

	double val = 1.0 / (double)count;

	for (auto compName : m_ComponentNames) {
		std::vector<double> vScales(m_Scales[compName].size(), val);
		std::vector<std::vector<double>> vScalesOrient;
		for (auto orient : m_Orientations) {
			vScalesOrient.push_back(vScales);
		}
		m_Alphas[compName] = vScalesOrient;
	}
}

void MultiscalePb::computeEdges() {
	printf("BlaBla1\n");
	std::vector<cv::Mat> orientGradientImages;
	printf("BlaBla2\n");
	for (unsigned int o = 0; o < m_Orientations.size(); ++o) {
		cv::Mat sumGrad = cv::Mat::zeros(m_OrigImage.size(), CV_64FC1);

		for (auto compName : m_ComponentNames) {
			for (unsigned int s = 0; s < m_Scales[compName].size(); ++s)
				cv::add(sumGrad, m_Alphas[compName][o][s] * m_GradientImages[compName][o][s], sumGrad);
		}
		orientGradientImages.push_back(sumGrad);
	}
	printf("BlaBla3\n");
	cv::Mat maxImage = cv::Mat::zeros(m_OrigImage.size(), CV_64FC1);
	printf("BlaBla4\n");
	for (int i = 0; i < m_OrigImage.rows; ++i) {
		for (int j = 0; j < m_OrigImage.cols; ++j) {
			double max = -1000000.9;
			for (unsigned int o = 0; o < m_Orientations.size(); ++o) {
				double val = orientGradientImages[o].at<double>(i, j);
				if (val > max)
					max = val;
			}
			maxImage.at<double>(i, j) = max;
		}
	}
	printf("BlaBla5\n");
	
	cv::Mat nonMaxSup = nonMaximumSuppression(orientGradientImages, maxImage);
	cv::normalize(nonMaxSup, m_GradImage, 0, 255, cv::NORM_MINMAX);
	printf("BlaBla6\n");
}

cv::Mat MultiscalePb::nonMaximumSuppression(const std::vector<cv::Mat> orientImgs, cv::Mat maxImage) {
	cv::Mat finalImage = cv::Mat::zeros(m_OrigImage.size(), CV_64FC1);

	for (int i = 0; i < m_OrigImage.rows; ++i) {
		for (int j = 0; j < m_OrigImage.cols; ++j) {
			int maxOrient = 0;
			double max = -10000000.0;
			for (unsigned int o = 0; o < m_Orientations.size(); ++o) {
				double val = orientImgs[o].at<double>(i, j);
				if (val > max) {
					max = val;
					maxOrient = o;
				}
			}

			if (maxOrient == 2) {
				//horizontal edge
				if (i == 0 || i == m_OrigImage.rows - 1) {
					finalImage.at<double>(i, j) = 0;
				} else {
					double val1 = maxImage.at<double>(i - 1, j);
					double val2 = maxImage.at<double>(i + 1, j);
					if (max >= val1 && max >= val2)
						finalImage.at<double>(i, j) = max;
					else
						finalImage.at<double>(i, j) = 0;
				}
			}

			if (maxOrient == 3) {
				if (i == 0 || j == m_OrigImage.cols - 1 || i == m_OrigImage.rows - 1 || j == 0) {
					finalImage.at<double>(i, j) = 0;
				} else {
					double val1 = maxImage.at<double>(i - 1, j + 1);
					double val2 = maxImage.at<double>(i + 1, j - 1);
					if (max >= val1 && max >= val2)
						finalImage.at<double>(i, j) = max;
					else
						finalImage.at<double>(i, j) = 0;
				}
			}

			if (maxOrient == 0) {
				if (j == 0 || j == m_OrigImage.cols - 1) {
					finalImage.at<double>(i, j) = 0;
				} else {
					double val1 = maxImage.at<double>(i , j + 1);
					double val2 = maxImage.at<double>(i , j - 1);
					if (max >= val1 && max >= val2)
						finalImage.at<double>(i, j) = max;
					else
						finalImage.at<double>(i, j) = 0;
				}
			}

			if (maxOrient == 1) {
				if (i == 0 || j == m_OrigImage.cols - 1 || i == m_OrigImage.rows - 1 || j == 0) {
					finalImage.at<double>(i, j) = 0;
				}
				else {
					double val1 = maxImage.at<double>(i - 1, j - 1);
					double val2 = maxImage.at<double>(i + 1, j + 1);
					if (max >= val1 && max >= val2)
						finalImage.at<double>(i, j) = max;
					else
						finalImage.at<double>(i, j) = 0;
				}
			}
		} // for j
	} // for i

	return finalImage;
}