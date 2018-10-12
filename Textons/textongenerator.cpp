#include "textongenerator.h"
#include <QDebug>
#include <QFileInfo>
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <cstdlib>
#include <ctime>

#include "gaussianfilter.h"
#include "gaussianfirstderivativefilter.h"
#include "gaussiansecondderivativefilter.h"


TextonGenerator::TextonGenerator() {
    std::srand(std::time(0));

    createFilterBanks();
    computeFilePaths();
    initClusterCenters();
}

void TextonGenerator::createFilterBanks() {
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
            TextonKernel* tk = new GaussianSecondDerivativeFilter( 5 + (k + 1) * 4, double(i) * 22.5, (k + 1) * sigma);
            tk->init();
            m_FilterBank.push_back(tk);
        }
    }

    TextonKernel* tk = new GaussianFilter(9, sigma);
    tk->init();
    m_FilterBank.push_back(tk);

    for (unsigned int i = 0; i < m_FilterBank.size(); ++i) {
        printf("Matrix %u\n", i);
        m_FilterBank[i]->printValues();
        printf("\n");
    }
}

void TextonGenerator::computeFilePaths() {
    QFileInfo qfi_dataPath(m_DataPath);
    if (!qfi_dataPath.exists() || !qfi_dataPath.isDir()) {
        printf("File does not exist\n");
        return;
    }
    QDir dir_dataPath(m_DataPath);
    QFileInfoList qfi_list_dataPath = dir_dataPath.entryInfoList();
    for (QFileInfo qfi_classDir : qfi_list_dataPath) {
        if (qfi_classDir.isDir()) {
            printf("%s\n", qPrintable(qfi_classDir.absoluteFilePath()));
            QDir dir_classDir(qfi_classDir.absoluteFilePath());
            QFileInfoList qfi_list_classDir = dir_classDir.entryInfoList();

            int count = 0;
            for (QFileInfo qfi_image : qfi_list_classDir) {
                if (qfi_image.isFile()) {
                    //printf("%s\n", qPrintable(qfi_image.absoluteFilePath()));
                    m_ImagesPaths.push_back(std::string(qfi_image.absoluteFilePath().toUtf8().constData()));
                    if (count < 2)
                        m_SmallSetImagesPaths.push_back(std::string(qfi_image.absoluteFilePath().toUtf8().constData()));
                    count++;
                }
            }
        }
    }
}

/*std::string imgPath = std::string(q1.absoluteFilePath().toUtf8().constData());
cv::Mat img = cv::imread(imgPath);
cv::Mat greyImg;
cv::cvtColor(img, greyImg, cv::COLOR_BGR2GRAY);*/

std::vector<cv::Mat> TextonGenerator::runFilterBankOnGrayscaleImage(const cv::Mat &greyImg) {
    std::vector<cv::Mat> retVal;
    for (auto f : m_FilterBank) {
        cv::Mat filtImg;
        cv::filter2D(greyImg, filtImg, -1, f->getKernel(), cv::Point(-1, -1), CV_16S);
        cv::Mat rescaled;
        cv::convertScaleAbs(filtImg, rescaled, 5.0);
        retVal.push_back(filtImg);
    }
    return retVal;
}

void TextonGenerator::initClusterCenters() {
    int imagesNo = static_cast<int>(m_SmallSetImagesPaths.size());
    for (int i = 0; i < m_ClusterNo; ++i) {
        int imgId = generateRandom(imagesNo);
        std::string imgPath = m_SmallSetImagesPaths[imgId];
        cv::Mat img = cv::imread(imgPath);
        cv::Mat greyImg;
        cv::cvtColor(img, greyImg, cv::COLOR_BGR2GRAY);
        int x = generateRandom(greyImg.cols);
        int y = generateRandom(greyImg.rows);
        std::vector<cv::Mat> filtImgs = runFilterBankOnGrayscaleImage(greyImg);
        Texton t = getTexton(filtImgs, x, y);
        m_ClusterCenters.push_back(t);
    }

    printf("ClusterCenters:\n");
    for (int i = 0; i < m_ClusterNo; ++i) {
        m_ClusterCenters[i].print();
    }
}

Texton TextonGenerator::getTexton(const std::vector<cv::Mat>& filtImgs, int x, int y) {
    Texton t(static_cast<int>(filtImgs.size()));
    for (unsigned int i = 0; i < filtImgs.size(); ++i) {
        t.setValueAtIdx(filtImgs[i].at<uchar>(y, x), i);
    }
    return t;
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

    std::vector<cv::Mat> filtImgs = runFilterBankOnGrayscaleImage(img);

    if (saveImages) {
        for (unsigned int i = 0; i < filtImgs.size(); ++i) {
            QString name = "D:\\ProjectsOpenCV\\Textons\\Tests\\test" + QString::number(imgIndex) + "_" + QString::number(i) + "_" + ".png";
            cv::imwrite(name.toUtf8().constData(), filtImgs[i]);
        }
    }
    imgIndex++;

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            Texton t = getTexton(filtImgs, j, i);
            int cIdx = getClosestClusterCenter(t);
            auto& v = kmeansData[cIdx];
            v.first = v.first + t;
            v.second = v.second + 1;
        }
    }

    return kmeansData;
}

int TextonGenerator::getClosestClusterCenter(const Texton& t) {
    int closest = 0;
    double minDist = t.dist(m_ClusterCenters[0]);
    for (int i = 1; i < m_ClusterNo; ++i) {
        double dist = t.dist(m_ClusterCenters[i]);
        if (dist < minDist) {
            minDist = dist;
            closest = i;
        }
    }
    return closest;
}

///@todo:
void TextonGenerator::runKMeansIteration() {
    std::vector<std::vector<std::pair<Texton, int>>> vKmeansData;
    int count = 0;
    int total = static_cast<int>(m_SmallSetImagesPaths.size());

    for (std::string imagePath : m_SmallSetImagesPaths) {
        printf("%s : %d/%d \n", imagePath.c_str(), count, total);
        cv::Mat img = cv::imread(imagePath);
        cv::Mat greyImg;
        cv::cvtColor(img, greyImg, cv::COLOR_BGR2GRAY);

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

void TextonGenerator::generateTestImages() {
    for (std::string imagePath : m_SmallSetImagesPaths) {
        printf("%s : \n", imagePath.c_str());
        cv::Mat img = cv::imread(imagePath);
        cv::Mat greyImg;
        cv::cvtColor(img, greyImg, cv::COLOR_BGR2GRAY);

        std::vector<std::pair<Texton, int>> kmeansData = runKMeansOnImage(greyImg, true);
    }
}

void TextonGenerator::execute() {
    writeClusterCentersToFile();
    for (int i = 0; i < m_IterationsNo; ++i) {
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
