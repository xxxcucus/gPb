#include "histoallocator.h"
#include <cstdlib>
#include <stdio.h>

HistoAllocator::HistoAllocator(int width, int height, int arcno, int scale)
	: m_Width(width), m_Height(height), m_ArcNo(arcno), m_Scale(scale),
	m_HistoCellSize(256 * 2 * m_ArcNo * (m_Width + 2 * m_Scale) * sizeof(unsigned int))
{
	//find how much memory is available
	int total = 0;
	int free = 0;
	m_LastCudaError = cudaMemGetInfo((size_t*)&free, (size_t*)&total);

	if (m_LastCudaError != cudaSuccess)
		return;

	int m_NoHistoChunks = free / 4 / m_HistoCellSize;
	printf("Allocating 2 chunks with %d histo cells\n", m_NoHistoChunks);

	m_LastCudaError = cudaMalloc((void**)&m_dChunk1, m_NoHistoChunks * m_HistoCellSize);
	if (m_LastCudaError != cudaSuccess) {
		return;
	}

	m_LastCudaError = cudaMemset(m_dChunk1, 0, m_NoHistoChunks * m_HistoCellSize);
	if (m_LastCudaError != cudaSuccess) {
		return;
	}

	m_LastCudaError = cudaMalloc((void**)&m_dChunk2, m_NoHistoChunks * m_HistoCellSize);
	if (m_LastCudaError != cudaSuccess) {
		return;
	}

	m_LastCudaError = cudaMemset(m_dChunk2, 0, m_NoHistoChunks * m_HistoCellSize);
	if (m_LastCudaError != cudaSuccess) {
		return;
	}

	//TODO: cudaMemset

	m_TopChunk1 = m_NoHistoChunks;
	m_BottomChunk1 = 0;
	m_TopChunk2 = 2 * m_NoHistoChunks;
	m_BottomChunk2 = m_NoHistoChunks;
}


HistoAllocator::~HistoAllocator() {
	cudaFree(m_dChunk1);
	cudaFree(m_dChunk2);
}

void HistoAllocator::setNewTopChunk() {	
	printf("SetNewTopChunk\n");
	m_LastCudaError = cudaFree(m_dChunk1);
	if (m_LastCudaError != cudaSuccess) {
		printf("BlaBla1\n");
		//return;
	}

	unsigned int* temp;

	m_LastCudaError = cudaMalloc((void**)&temp, m_NoHistoChunks * m_HistoCellSize);
	if (m_LastCudaError != cudaSuccess) {
		printf("BlaBla2\n");
		return;
	}

	m_LastCudaError = cudaMemset(temp, 0, m_NoHistoChunks * m_HistoCellSize);
	if (m_LastCudaError != cudaSuccess) {
		printf("BlaBla3\n");
		return;
	}

	//TODO: is this really working??
	m_dChunk1 = m_dChunk2;
	m_dChunk2 = temp;

	m_BottomChunk1 = m_BottomChunk2;
	m_TopChunk1 = m_TopChunk2;
	m_BottomChunk2 = m_TopChunk1;
	m_TopChunk2 = m_BottomChunk2 + m_TopChunk1 - m_BottomChunk1;
}