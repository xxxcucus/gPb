#ifndef __HISTO_ALLOCATOR__
#define __HISTO_ALLOCATOR__

#include <cuda.h>
#include <cuda_runtime.h>

/**
* Class that manages memory for histograms.
* The available memory is divided into two memory chunks denoted with index 1 and 2.
* Each chunk is made of m_NoHistoChunks blocks of memory, each of this corresponding to the
* memory required to save the histograms for one row of the image.
* The value of the m_NoHistoChunks is computed based on the available memory in the CUDA device,
* the scale of the Pb filter, and the number of rows computed in one step by the CUDA kernels.
* Initially 
*  	m_TopChunk1 = m_NoHistoChunks;
*	m_BottomChunk1 = 0;
*	m_TopChunk2 = 2 * m_NoHistoChunks;
*	m_BottomChunk2 = m_NoHistoChunks;
* and the histograms data is saved in the first chunk.
* When data for chunk1 is not available anymore, the memory is resetted to 0 and m_TopChunk1 and m_BottomChunk1
* are set to point to the next m_NoHistoChunks of rows of histogram memory.
* Then the lowest chunk will be the second chunk and all the work will be done with it until it will not be needed
* anymore. .... and so forth and so forth ..
*/


class HistoAllocator {

public: 
	HistoAllocator(int width, int height, int arcno, int scale, int step);
	~HistoAllocator();

	bool wasError() {
		return m_LastCudaError != cudaSuccess;
	}

	cudaError_t getError() {
		return m_LastCudaError;
	}

	void setNewTopChunk();

//TODO: to make this private
public:
	
	//image width
	const int m_Width;
	//image height
	const int m_Height;
	//number of orientations used by the Pb algorithm
	const int m_ArcNo;
	//scale of the Pb algorithm
	const int m_Scale;
	//number of rows in the image calculated concurrently by CUDA the kernel functions
	const int m_Step;

	//how much data is required for one row of histograms in the cuda implementation
	size_t m_HistoCellSize; 
	//number of rows in one histogram rows stored in one histogram chunk
	size_t m_NoHistoChunks = 2 * m_Scale;

	//histograms in the device
	unsigned int** m_dHistograms;
	//histograms in host
	unsigned int** m_hHistograms;

	//index of last row pointed by the first chunk
	int m_TopChunk1 = 0;
	//index of the first row pointed by the first chunk
	int m_BottomChunk1 = 0;
	//index of the last row pointed by the second chunk
	int m_TopChunk2 = 0;
	//index of the first row pointed by the first chunk
	int m_BottomChunk2 = 0;

	cudaError_t m_LastCudaError = cudaSuccess;
};


#endif