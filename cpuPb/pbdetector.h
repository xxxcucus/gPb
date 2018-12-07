#ifndef PBDETECTOR_H
#define PBDETECTOR_H
#include "opencv2/opencv.hpp"
#include <QString>
#include <map>
#include <queue>

#include "discinversemasks.h"

/*
 * We have to define:
 *      - masks for 8 subregions of a disk with parametrisable radius centered at any point
 *      - data structure encompasing histograms for 8 orientations, for the two half discs
 *      - optimum generation of histograms for an image, reverted masks for quick implementation
 */


//histogram of image values
typedef std::map<int, int> Histo;
//for every pixel in image save an array of histograms, each for one orientation
//first std::vector contains rows, second std::vector contains
//elements on each column in the chosen row, the last std::vector contains one histogram for each
//boundary orientation
typedef std::vector<std::vector<std::vector<Histo>>> HistoVect;

/**
 * @brief The PbDetector class implements the probability of boundary contour detector
 */
class PbDetector
{   
public:
    PbDetector(int scale, const cv::Mat& image);
    void calculateGradients();
    inline cv::Mat& getGradientImage(int i) { return m_GradientImages[i]; }

private:
    /**
     * @brief PbDetector::calculateGradients - Computes Chi-Square distance between the two half discs
     * for the calculated histograms for an entire row of the input image and all the 4 orientations
     * @param vMaps - the already calculated histograms
     * @param (index - m_Scale)- row of histograms to be processed
     */
    void calculateGradients(const HistoVect& vMaps, int index);
    void addToHistoMaps(HistoVect& vMaps, int val, int i, int j);
    /**
     * @brief PbDetector::deleteFromHistoMaps - Deletes the computed histograms from memory for the already calculated
     * gradients
     * @param vMaps - the already calculated histograms
     * @param (index - m_Scale) - row of histograms to be processed
     */
    void deleteFromHistoMaps(HistoVect& vMaps, int index);
    //void calculateGradients(int i);

    /**
     * @brief chisquare - computes chisquare difference between two histograms
     * @param histo1 - first histogram
     * @param histo2 - second histogram
     */
    double chisquare(const Histo& histo1, const Histo& histo2);

    /**
     * @brief insertInHisto - add value to histogram
     * @param histo - concerned histogram
     * @param val - value to be added
     */
    void insertInHisto(Histo& histo, int val);

    /**
     * @brief initializeHistoRange - creates empty histograms for rows of image
     * between start and stop.
     * @param vMaps
     * @param start
     * @param stop
     */
    void initializeHistoRange(HistoVect& vMaps, int start, int stop);


private:
    const cv::Mat& m_SingleChannelImage;
    std::vector<cv::Mat> m_GradientImages;
    //the size of the half discs used by the detector
    int m_Scale;
    //the half disc masks around a center point
    DiscInverseMasks* m_Masks = nullptr;
    int m_ArcNo = 4;
};

#endif // PBDETECTOR_H
