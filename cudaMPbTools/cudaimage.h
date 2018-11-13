#ifndef _CUDA_Image_
#define _CUDA_Image_

#include "discinversemasks.h"
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void calculateGradients(int row, double* dGradientImages, unsigned int** dHistograms, int image_width, int image_height, int scale, int arcno);
__global__ void calcHisto(int row, unsigned char* dSourceImage, struct CVector* dHalfDiscInfluencePoints, int totalHalfInfluencePoints, unsigned int** dHistograms, int image_width, int image_height, int scale, int arcno);
__device__ void addToHistoArray(struct CVector* dHalfDiscInfluencePoints, int totalHalfInfluencePoints, unsigned int** dHistograms, int image_width, int image_height, int scale, int arcno, int val, int i, int j);
__device__ double chisquare(unsigned int* histo1, unsigned int* histo2);

class CudaImage {
public:
	CudaImage(unsigned char* image_data, int image_width, int image_height, int scale);
	~CudaImage();

	bool wasSuccessfullyCreated() {
		return m_FullyInitialized;
	}

	void execute();

	/**
	* returns the gradient image corresponding to the index arc
	*/
	double* getGradientImage(int index) {
		return m_hGradientImages + index * m_Width * m_Height;
	}

private:
	
    /**
    * Allocates image on the GPU, pads it with zeros, and copies from host to gpu the image data. 
    */
	bool copyImageToGPU(unsigned char* image_data);
    /**
    * Creates 2D histogram array in the GPU.
    */
	bool create2DHistoArray();
    /**
    * Allocates memory for the gradient images.
    */
	bool createGradientImages();



    /**
     * Creates empty histograms on the GPU
     * for rows of image between start and stop.
     * @param start
     * @param stop
     */
    bool initializeHistoRange(int start, int stop);	

	/**
	* Creates empty histograms on the GPU
	* for new row and deletes histograms 
	* which are not used anymore
	* such that memory is used efficiently
	* @param index
	*/
	void deleteFromHistoMaps(int index);

    /**
     * Copies m_Masks->getHalfDiscInfluencePoints()
     * to the GPU
     */
    bool initializeInfluencePoints();


private:
	unsigned char* m_dSourceImage; //image on the GPU
	double* m_dGradientImages; //the result images
	double* m_hGradientImages = nullptr;

	unsigned int** m_dHistograms;
	struct CVector* m_dHalfDiscInfluencePoints;
	struct CVector* m_hHalfDiscInfluencePoints = nullptr;
	int m_TotalHalfInfluencePoints = 0;

	int m_Width = 0;
	int m_Height = 0;
	int m_Scale = 0;

	cudaError_t m_LastCudaError;

	int m_ArcNo = 4;
	bool m_FullyInitialized = false;

	//the half disc masks around a center point
    DiscInverseMasks* m_Masks = nullptr;

};


#endif

