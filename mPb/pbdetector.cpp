#include "pbdetector.h"
#include "discinversemasks.h"
#include <QDebug>
#include <QTime>

PbDetector::PbDetector(int scale, const cv::Mat& image): m_SingleChannelImage(image), m_Scale(scale) {
    m_Masks = new DiscInverseMasks(scale);
    for (int i = 0; i < m_ArcNo; ++i) {
        cv::Mat gradientImage = cv::Mat::zeros(image.size(), CV_64FC1);
        m_GradientImages.push_back(gradientImage);
    }
}

void PbDetector::calculateGradients() {
    qDebug() << "Build empty histograms";
    HistoVect histograms;
    initializeHistoRange(histograms, 0, m_Scale + 1);

    //padded image
    cv::Mat bufImg(m_SingleChannelImage.rows + 2 * m_Scale, m_SingleChannelImage.cols + 2 * m_Scale, m_SingleChannelImage.type());
    cv::copyMakeBorder(m_SingleChannelImage, bufImg, m_Scale, m_Scale, m_Scale, m_Scale, cv::BORDER_REPLICATE);

    qDebug() << "Compute gradients on " << bufImg.rows << " " << bufImg.cols << QTime::currentTime().toString("hh:mm:ss:zzz");

    for (int i = 0; i < m_SingleChannelImage.rows + 2 * m_Scale; ++i) {
        //calculate histogram differences for line i - m_Radius * 2
        //qDebug() << "Compute step " << i << QTime::currentTime().toString("hh:mm:ss:zzz");
        for (int j = 0; j < m_SingleChannelImage.cols + 2 * m_Scale; ++j) {
            int val = bufImg.at<unsigned char>(i, j);
            //with the point (i,j) with value val, update all histograms which contain this data point
            addToHistoMaps(histograms, val, i, j);
        }
        //qDebug() << "Gradients step " << i << QTime::currentTime().toString("hh:mm:ss:zzz");
        calculateGradients(histograms, i);
        //qDebug() << "Delete step " << i << QTime::currentTime().toString("hh:mm:ss:zzz");
        //to optimized the memory used only m_Scale + 1 rows are kept in memory
        deleteFromHistoMaps(histograms, i);
    }
    qDebug() << "Compute gradients has finished " << QTime::currentTime().toString("hh:mm:ss:zzz");
}

void PbDetector::addToHistoMaps(HistoVect& vMaps, int val, int i, int j) {
    ///origin relative to i,j
    ///in each of the 8 orientations in which half it belongs
    /// std::pair<int,int>
    std::vector<std::vector<int>>& neighb = m_Masks->getHalfDiscInfluencePoints();
    for (auto n : neighb) {
        if ((n[0] + i) < 0 || (n[0] + i) >= m_SingleChannelImage.rows + 2 * m_Scale)
            continue;
        if ((n[1] + j) < 0 || (n[1] + j) >= m_SingleChannelImage.cols + 2 * m_Scale)
            continue;

        //qDebug() << "Compute at " << i << " with " << (n[0] + i) << " size " << vMaps[n[0] + i].size();
        if (int(vMaps[n[0] + i].size()) != m_SingleChannelImage.cols + 2 * m_Scale)
            exit(1);

        std::vector<Histo>& vHist = vMaps[n[0] + i][n[1] + j];
        for (unsigned int k = 2; k < n.size(); ++k) {
            if (n[k] > 2 * m_ArcNo)
                continue;
            insertInHisto(vHist[n[k]], val);
        }
    }
}

void PbDetector::deleteFromHistoMaps(HistoVect& vMaps, int index) {
    //add a new row in the vMaps
    if (index + m_Scale + 1 < m_SingleChannelImage.rows + 2 * m_Scale) {
        initializeHistoRange(vMaps, index + m_Scale + 1, index + m_Scale + 2);
        //qDebug() << "Initialize " << index + m_Scale + 1;
    }
    //delete row which was already analyzed from vMaps
    if (index >=  m_Scale + 1) {
        //qDebug() << "Delete " << index - m_Scale;
        vMaps[index - m_Scale - 1].clear();
    }
}

void PbDetector::calculateGradients(const HistoVect& vMaps, int index) {
    if (index < 2 * m_Scale)
        return;

    for (int j = m_Scale; j < m_SingleChannelImage.cols + m_Scale; ++j) {
        //for each of the 4 possible half disc divisions
        //qDebug() << "index " << index << " " << j;
        const std::vector<Histo>& vHist = vMaps[index - m_Scale][j];
        if (int(vHist.size()) != 2 * m_ArcNo)
            continue;

        for (int i = 0; i < m_ArcNo; ++i) {
            const Histo& histo1 = vHist[i];
            const Histo& histo2 = vHist[i + m_ArcNo];
            double grad = chisquare(histo1, histo2);
            m_GradientImages[i].at<double>(index - 2 * m_Scale, j - m_Scale) = grad;
        }
    }
}

///@todo: complexity?
double PbDetector::chisquare(const Histo& histo1, const Histo& histo2) {
    double retVal = 0.0;

    for (auto h : histo1) {
        auto it = histo2.find(h.first);
        if (it == histo2.end())
            retVal += h.second;
        else
            retVal += double(it->second - h.second) * double(it->second - h.second) / double(it->second + h.second);
    }

    for (auto h : histo2) {
        auto it = histo1.find(h.first);
        if (it == histo1.end())
            retVal += h.second;
    }

    return retVal;
}

void PbDetector::insertInHisto(Histo& histo, int val) {
    auto it = histo.find(val);
    if (it == histo.end()) {
        histo[val] = 1;
    } else {
        histo[val]++;
    }
}


void PbDetector::initializeHistoRange(HistoVect& vMaps, int start, int stop) {
    std::vector<Histo> vHist;
    for (int i = 0; i < m_ArcNo; ++i)
        vHist.push_back(std::map<int, int>());
    for (int i = start; i < stop; ++i) {
        //qDebug() << "Initialized row " << i;
        std::vector<std::vector<Histo>> vvHist;
        for (int j = 0; j < m_SingleChannelImage.cols + 2 * m_Scale; ++j)
            vvHist.push_back(vHist);
        vMaps.push_back(vvHist);
    }
}
