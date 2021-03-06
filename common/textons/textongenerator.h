#ifndef TEXTONGENERATOR_H
#define TEXTONGENERATOR_H
#include "opencv2/opencv.hpp"

#include "textonkernel.h"
#include "texton.h"
#include "filterbank.h"
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
    TextonGenerator(FilterBank& filterBank, const QString& path);
    /**
     * @brief: runs runKMeansIteration m_IterationNo times and writes the results after each step to file
     */
    void execute();

	void setDataPath(const QString& path) {
		m_DataPath = path;
	}

	void setTargetPath(const QString& path) {
		m_TargetPath = path;
	}

private:
    
    /**
     * @brief saves the file paths that will be used for the texton generation
     */
    void computeFilePaths();
    /**
     * @brief initialize the cluster centers for KMeans clustering of the textons
     */
    void initClusterCenters();

    void writeClusterCentersToFile(const QString& path);

    /**
     * @brief runKMeansOnImage - performs kmeans clustering and saves a vector of textons representing
     * the sums of all textons closest to each cluster center as well as the number of textons in the sum
     * @param img
     */
    std::vector<std::pair<Texton, int>> runKMeansOnImage(const cv::Mat& img, bool saveImages);

    /**
     * @brief generateRandom - generates random integer between 0 and maxVal - 1
     * @param maxVal
     * @return
     */
    int generateRandom(int maxVal);

    /**
     * @brief runKMeansIteration - reads all the images, and runs runKMeansOnImage, gathers the results for all images
     * and recalculates the cluster center positions
     */
    void runKMeansIteration();


private:
    QString m_DataPath = "../textures/";
	QString m_TargetPath = "textons.txt";
    QString m_ReprTextonsPath = "cluster_centers.txt";

	FilterBank& m_FilterBank;

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
