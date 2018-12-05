#include "textongenerator.h"
#include <QDebug>
#include <QFileInfo>
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <cstdlib>
#include <ctime>
#include "textontools.h"

TextonGenerator::TextonGenerator(FilterBank& filterBank) : m_FilterBank(filterBank) {
    std::srand(std::time(0));
	printf("Compute file paths \n");
	computeFilePaths();
	printf("Init cluster centers \n");
	initClusterCenters();
}

void TextonGenerator::computeFilePaths() {
    QFileInfo qfi_dataPath(m_DataPath);
    if (!qfi_dataPath.exists() || !qfi_dataPath.isDir()) {
        printf("File does not exist\n");
        return;
    }
    QDir dir_dataPath(m_DataPath);
	QStringList filters;
	printf("Searching in path %s\n", qPrintable(dir_dataPath.canonicalPath()));
	QFileInfoList qfi_list_dataPath = dir_dataPath.entryInfoList(filters);
    for (QFileInfo qfi_elem : qfi_list_dataPath) {
        if (qfi_elem.isDir()) {
            printf("1 %s\n", qPrintable(qfi_elem.absoluteFilePath()));
            QDir dir_classDir(qfi_elem.absoluteFilePath());
            QFileInfoList qfi_list_classDir = dir_classDir.entryInfoList(filters, QDir::NoDot | QDir::NoDotDot);

            int count = 0;
            for (QFileInfo qfi_image : qfi_list_classDir) {
                if (qfi_image.isFile()) {
                    printf("11 %s\n", qPrintable(qfi_image.absoluteFilePath()));
                    m_ImagesPaths.push_back(qfi_image.absoluteFilePath().toUtf8().constData());
                    //if (count < 2)
                    m_SmallSetImagesPaths.push_back(qfi_image.absoluteFilePath().toUtf8().constData());
                    count++;
                }
            }
		} else if (qfi_elem.isFile()) {
			printf("2 %s\n", qPrintable(qfi_elem.absoluteFilePath()));
			m_ImagesPaths.push_back(qfi_elem.absoluteFilePath().toUtf8().constData());
			m_SmallSetImagesPaths.push_back(qfi_elem.absoluteFilePath().toUtf8().constData());
		}
    }
}

void TextonGenerator::initClusterCenters() {
    int imagesNo = static_cast<int>(m_SmallSetImagesPaths.size());
    for (int i = 0; i < m_ClusterNo; ++i) {
        int imgId = generateRandom(imagesNo);
        std::string imgPath = m_SmallSetImagesPaths[imgId];
        cv::Mat greyImg = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
        int x = generateRandom(greyImg.cols);
        int y = generateRandom(greyImg.rows);
        std::vector<cv::Mat> filtImgs = m_FilterBank.runOnGrayScaleImage(greyImg);
        Texton t = TextonTools::getTexton(filtImgs, x, y);
        m_ClusterCenters.push_back(t);
    }

    printf("ClusterCenters:\n");
    for (int i = 0; i < m_ClusterNo; ++i) {
        m_ClusterCenters[i].print();
    }
}

int TextonGenerator::generateRandom(int maxVal) {
    return (std::rand() % maxVal);
}

std::vector<std::pair<Texton, int>> TextonGenerator::runKMeansOnImage(const cv::Mat& img, bool saveImages) {
    static int imgIndex = 0;
    std::vector<std::pair<Texton, int>> kmeansData;
    for (int i = 0; i < m_ClusterNo; ++i) {
        kmeansData.push_back(std::make_pair(Texton(), 0));
    }

    std::vector<cv::Mat> filtImgs = m_FilterBank.runOnGrayScaleImage(img);

    if (saveImages) {
        for (unsigned int i = 0; i < filtImgs.size(); ++i) {
            QString name = "test" + QString::number(imgIndex) + "_" + QString::number(i) + "_" + ".png";
            cv::imwrite(name.toUtf8().constData(), filtImgs[i]);
        }
    }
    imgIndex++;

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            Texton t = TextonTools::getTexton(filtImgs, j, i);
            int cIdx = TextonTools::getClosestClusterCenter(t, m_ClusterCenters);
            auto& v = kmeansData[cIdx];
            v.first = v.first + t;
            v.second = v.second + 1;
        }
    }

    return kmeansData;
}

///@todo:
void TextonGenerator::runKMeansIteration() {
    std::vector<std::vector<std::pair<Texton, int>>> vKmeansData;
    int count = 0;
    int total = static_cast<int>(m_SmallSetImagesPaths.size());

    for (std::string imagePath : m_SmallSetImagesPaths) {
        printf("%s : %d/%d \n", imagePath.c_str(), count, total);
        cv::Mat greyImg = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

        std::vector<std::pair<Texton, int>> kmeansData = runKMeansOnImage(greyImg, false);
        vKmeansData.push_back(kmeansData);
        count++;
    }

    std::vector<int> counters(m_ClusterNo, 0.0);
    for (auto dImg : vKmeansData) {
        for (unsigned int i = 0; i < dImg.size(); ++i) {
            counters[i] += dImg[i].second;
        }
    }

    std::vector<Texton> sums(m_ClusterNo, Texton());
    for (auto dImg : vKmeansData) {
        for (unsigned int i = 0; i < dImg.size(); ++i) {
            if (counters[i] < 0.0001) {
                qDebug() << "Counters " << i << " is " << counters[i];
                sums[i] = 0.0;
            }
            sums[i] = sums[i] + dImg[i].first / double(counters[i]);
        }
    }

    printf("ClusterCenters:\n");
    for (int i = 0; i < m_ClusterNo; ++i) {
        m_ClusterCenters[i] = sums[i];
        m_ClusterCenters[i].print();
    }
}

void TextonGenerator::execute() {
	printf("Execute \n");
    writeClusterCentersToFile();
    for (int i = 0; i < m_IterationsNo; ++i) {
		printf("Iteration %d\n", i + 1);
		runKMeansIteration();
        writeClusterCentersToFile();
    }
}

void TextonGenerator::writeClusterCentersToFile() {
    QFile file(m_ReprTextonsPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Append)) {
        qDebug()  << "File could not be opened";
        return;
    }

    QTextStream stream(&file);

    stream << endl;
    stream << endl;
    for (int i = 0; i < m_ClusterNo; ++i) {
        stream << m_ClusterCenters[i].toString() << endl;
    }
    stream.flush();

    file.close();
}
