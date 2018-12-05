#include "textontools.h"

#include <fstream>
#include <QString>
#include <QStringList>

#include "filterbank.h"

//TODO: error checking
bool TextonTools::readFromTextonsFile(const std::string& path, std::vector<Texton>& textons) {
	std::ifstream ifs(path);

	if (ifs.rdstate() & std::ifstream::failbit) {
#ifdef _DEBUG
		printf("Failed to open config file %s\n", path.c_str());
#endif	
		return false;
	}

	while ((ifs.rdstate() & std::ifstream::eofbit) == 0) {
		std::string line;
		std::getline(ifs, line);
#ifdef _DEBUG
		printf("%s\n", line.c_str());
#endif
		QString qLine = QString(line.c_str());
		QStringList strComps = qLine.split(" ");
		
		Texton t;

		int count = 0;
		for (auto strComp : strComps) {
			double dComp = strComp.toDouble();
			if (count < 0 || count >= 17)
				break;
			t.setValueAtIdx(dComp, count);
			count++;
		}

		textons.push_back(t);
	}

	return true;
}

//image should be CV_UC1
bool TextonTools::convertToTextonImage(cv::Mat img, const std::vector<Texton> quantTextons, cv::Mat& result) {
	result = cv::Mat::zeros(img.size(), CV_8UC1);
	
	FilterBank filterBank;
	std::vector<cv::Mat> filtImages = filterBank.runOnGrayScaleImage(img);

	//todo: implement cuda version if necessary
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			Texton t = getTexton(filtImages, j, i);
			int idx = getClosestClusterCenter(t, quantTextons);
			result.at<uchar>(i, j) = idx;
		}
	}
		
	return true;
}

//TODO: to add error handling
Texton TextonTools::getTexton(const std::vector<cv::Mat>& filtImgs, int x, int y) {
	Texton t(static_cast<int>(filtImgs.size()));
	for (unsigned int i = 0; i < filtImgs.size(); ++i) {
		cv::Mat img = filtImgs[i];
		if (y < 0 || y >= img.rows)
			continue;
		if (x < 0 || x >= img.cols)
			continue;
		
		t.setValueAtIdx(filtImgs[i].at<uchar>(y, x), i);
	}
	return t;
}

//TODO: to add error handling
int TextonTools::getClosestClusterCenter(const Texton& t, const std::vector<Texton>& cluster) {
	if (cluster.empty())
		return 0;
	int closest = 0;
	double minDist = t.dist(cluster[0]);
	for (int i = 1; i < cluster.size(); ++i) {
		double dist = t.dist(cluster[i]);
		if (dist < minDist) {
			minDist = dist;
			closest = i;
		}
	}
	return closest;
}