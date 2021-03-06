#ifndef _CUDA_PbDetector_
#define _CUDA_PbDetector_


#include <cuda.h>
#include <cuda_runtime.h>
#include "discinversemasks.h"
#include "histoallocator.h"

__global__ void calculateGradients(int row, int row_count, double* dGradientImages, unsigned int** dHistograms, int bottomChunk1, int bottomChunk2, int topChunk1, int topChunk2, int image_width, int image_height, int scale, int arcno);
__global__ void calcHisto(int row, int row_count, unsigned char* dSourceImage, int* dHalfDiscInfluencePoints, int totalHalfInfluencePoints, unsigned int** dHistograms, int bottomChunk1, int bottomChunk2, int topChunk1, int topChunk2, int image_width, int image_height, int scale, int arcno);
__device__ void addToHistoArray(int* dHalfDiscInfluencePoints, int totalHalfInfluencePoints, unsigned int** dHistograms, int bottomChunk1, int bottomChunk2, int topChunk1, int topChunk2, int image_width, int image_height, int scale, int arcno, int val, int i, int j);
__device__ unsigned int* getHistoPointer(int row, int col, unsigned int** dHistograms, int bottomChunk1, int bottomChunk2, int topChunk1, int topChunk2, int image_width, int scale, int arcno);

class CudaPbDetector {
public:
	CudaPbDetector(unsigned char* image_data, int image_width, int image_height, int scale);
	~CudaPbDetector();

	bool wasSuccessfullyCreated() {
		return m_FullyInitialized;
	}

	const char* getErrorString() {
		return cudaGetErrorString(m_LastCudaError);
	}

	bool executeChunk();

	/**
	* returns the gradient image corresponding to the index arc
	*/
	double* getGradientImage(int index) {
		return m_hGradientImages + index * m_Width * m_Height;
	}

private:
	
	void readDeviceProperties();

    /**
    * Allocates image on the GPU, pads it with zeros, and copies from host to gpu the image data. 
    */
	bool copyImageToGPU(unsigned char* image_data);

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
	bool updateHistoBuffer(int step, int index);

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
	unsigned int** m_hHistograms;
	int* m_dHalfDiscInfluencePoints;
	int* m_hHalfDiscInfluencePoints = nullptr;
	int m_TotalHalfInfluencePoints = 0;

	int m_Width = 0;
	int m_Height = 0;
	int m_Scale = 0;

	int m_Step = 7;
	int m_NoThreads = 256;

	cudaError_t m_LastCudaError;

	int m_ArcNo = 4;
	bool m_FullyInitialized = false;

	//the half disc masks around a center point
    DiscInverseMasks* m_Masks = nullptr;

	HistoAllocator* m_HistoAllocator = nullptr;

	size_t m_SharedMemoryPerBlock;
};


#endif

