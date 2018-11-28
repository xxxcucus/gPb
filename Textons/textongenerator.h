#ifndef TEXTONGENERATOR_H
#define TEXTONGENERATOR_H
#include "opencv2/opencv.hpp"

#include "textonkernel.h"
#include "texton.h"
#include <QString>

/**
 * @brief Should be a command line tool to compute textons given a folder with images.
 * The program calculates the results of the filters on all images and runs
 * KMeans clustering on the multidimensional data corresponding to the points
 * Results should be saved in a text file.
 * Set of filter banks should be configurable.
 * Number of textons should be received as parameter
 * Path to images folder should be received as parameter
 */

class TextonGenerator
{
public:
    TextonGenerator();
    /**
     * @brief calculates filter bank response on the images in m_SmallSetImagesPaths
     */
    void generateTestImages();
    /**
     * @brief: runs runKMeansIteration m_IterationNo times and writes the results after each step to file
     */
    void execute();

	void setDataPath(const QString& path) {
		m_DataPath = path;
	}


private:
    void createFilterBanks();
    /****************
     * @brief creates the file paths that will be used for the texton generation
     * is not necessary maybe to use so many files
     */
    void computeFilePaths();
    /**
     * @brief initialize the most representative textons
     */
    void initClusterCenters();

    void writeClusterCentersToFile();

    /**
     * @brief runKMeansOnImage - performs kmeans clustering and saves a vector of textons representing
     * the sums of all textons closest to each cluster center as well as the number of textons in the sum
     * @param img
     */
    std::vector<std::pair<Texton, int>> runKMeansOnImage(const cv::Mat& img, bool saveImages);

    /**
     * @brief runFilterBankOnGrayscaleImage - run on the filters on the image to obtain a multidimensional filtered image
     * @param img
     * @return
     */
    std::vector<cv::Mat> runFilterBankOnGrayscaleImage(const cv::Mat& img);

    /**
     * @brief generateRandom - generates random integer between 0 and maxVal - 1
     * @param maxVal
     * @return
     */
    int generateRandom(int maxVal);

    /**
     * @brief getTexton - gets a texton from a set of filtered images
     * @param filtImgs
     * @param x - position in the images the texton is calculated for
     * @param y
     * @return
     */
    Texton getTexton(const std::vector<cv::Mat>& filtImgs, int x, int y);
    int getClosestClusterCenter(const Texton& t);

    /**
     * @brief runKMeansIteration - reads all the images, and runs runKMeansOnImage, gathers the results for all images
     * and recalculates the cluster center positions
     */
    void runKMeansIteration();


private:
    QString m_DataPath = "data";
    QString m_ReprTextonsPath = "cluster_centers.txt";

    //filters used to compute the textons
    std::vector<TextonKernel*> m_FilterBank;
    //images used to compute the most representative textons
    std::vector<std::string> m_ImagesPaths;
    std::vector<std::string> m_SmallSetImagesPaths;

    //number of representative textons to be computed
    int m_ClusterNo = 32;
    //the representative textons
    std::vector<Texton> m_ClusterCenters;
    //number of times kmeans runs
    int m_IterationsNo = 10;

};

#endif // TEXTONGENERATOR_H
