#include "textontools.h"

#include <fstream>
#include <QString>
#include <QStringList>

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
bool TextonTools::convertToTextonImage(cv::Mat img, cv::Mat result) {
	


}