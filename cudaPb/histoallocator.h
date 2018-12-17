#ifndef __HISTO_ALLOCATOR__
#define __HISTO_ALLOCATOR__

#include <cuda.h>
#include <cuda_runtime.h>

class HistoAllocator {

public: 
	HistoAllocator(int width, int height, int arcno, int scale);
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
	const int m_Width;
	const int m_Height;
	const int m_ArcNo;
	const int m_Scale;

	size_t m_HistoCellSize; 
	int m_NoHistoChunks = 2 * m_Scale;

	unsigned int** m_dHistograms;
	unsigned int** m_hHistograms;

	int m_TopChunk1 = 0;
	int m_BottomChunk1 = 0;
	int m_TopChunk2 = 0;
	int m_BottomChunk2 = 0;

	cudaError_t m_LastCudaError = cudaSuccess;
};


#endif