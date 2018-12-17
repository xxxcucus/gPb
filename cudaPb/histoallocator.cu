#include "histoallocator.h"
#include <cstdlib>
#include <stdio.h>

HistoAllocator::HistoAllocator(int width, int height, int arcno, int scale): 
	m_Width(width), m_Height(height), m_ArcNo(arcno), m_Scale(scale) {

	m_HistoCellSize = 256 * 2 * m_ArcNo * (m_Width + 2 * m_Scale) * sizeof(unsigned int);
	//find how much memory is available
	size_t total = 0;
	size_t free = 0;
	m_LastCudaError = cudaMemGetInfo(&free, &total);

	if (m_LastCudaError != cudaSuccess) 
		return;
	

	size_t m_NoHistoChunks = free / 4 / m_HistoCellSize;
	printf("Allocating 2 chunks with %zu histo cells. Free %zu Total %zu\n", m_NoHistoChunks, free, total);
	printf("Arcno %d Width %d Scale %d\n", m_ArcNo, m_Width, m_Scale);
	printf("Cell size %zu %llu\n", m_HistoCellSize, sizeof(unsigned int));

	//preparing histograms
	m_LastCudaError = cudaMalloc((void**)&m_dHistograms, 2 * sizeof(unsigned int*));
	printf("BlaBla10\n");

	if (m_LastCudaError != cudaSuccess) {
		printf("BlaBla1\n");
		return;
	}

	printf("BlaBla11\n");
	m_hHistograms = (unsigned int**)malloc(2 * sizeof(unsigned int*));
/*	if (m_LastCudaError != cudaSuccess) {
		printf("BlaBla1-1\n");
		return;
	}*/
	printf("BlaBla12 - %d\n", m_NoHistoChunks * m_HistoCellSize);
	for (int i = 0; i < 2; ++i) {
		m_LastCudaError = cudaMalloc((void**)&m_hHistograms[i], m_NoHistoChunks * m_HistoCellSize);
		//printf("Alloc %d\n", i);
		if (m_LastCudaError != cudaSuccess) {
			printf("cudaMalloc error 1: %d\n", i);
			return;
		}

		cudaMemcpy(m_dHistograms, m_hHistograms, 2 * sizeof(unsigned int*), cudaMemcpyHostToDevice);
		if (m_LastCudaError != cudaSuccess) {
			printf("cudaMemcpy error 1\n");
			return;
		}
	}

	printf("BlaBla13\n");
	m_LastCudaError = cudaMemset(m_hHistograms[0], 0, m_NoHistoChunks * m_HistoCellSize);
	if (m_LastCudaError != cudaSuccess) {
		printf("cudaMemset error 1\n");
		return;
	}

	printf("BlaBla14\n");
	m_LastCudaError = cudaMemset(m_hHistograms[1], 0, m_NoHistoChunks * m_HistoCellSize);
	if (m_LastCudaError != cudaSuccess) {
		printf("cudaMemset error 2\n");
		return;
	}

	printf("BlaBla15\n");
	m_TopChunk1 = int(m_NoHistoChunks);
	m_BottomChunk1 = 0;
	m_TopChunk2 = 2 * int(m_NoHistoChunks);
	m_BottomChunk2 = int(m_NoHistoChunks);
}


HistoAllocator::~HistoAllocator() {
	cudaFree(m_hHistograms[0]);
	cudaFree(m_hHistograms[1]);
	cudaFree(m_dHistograms);
	free(m_hHistograms);
}

void HistoAllocator::setNewTopChunk() {	
	printf("SetNewTopChunk %d-%d-%d-%d\n", m_BottomChunk1, m_TopChunk1, m_BottomChunk2, m_TopChunk2);
	/*m_LastCudaError = cudaFree(m_dChunk1);
	if (m_LastCudaError != cudaSuccess) {
		printf("BlaBla1\n");
		//return;
	}*/

	/*unsigned int* temp;

	m_LastCudaError = cudaMalloc((void**)&temp, m_NoHistoChunks * m_HistoCellSize);
	if (m_LastCudaError != cudaSuccess) {
		printf("BlaBla2\n");
		return;
	}*/

	if (m_BottomChunk1 < m_BottomChunk2) {
		m_LastCudaError = cudaMemset(m_hHistograms[0], 0, m_NoHistoChunks * m_HistoCellSize);
		if (m_LastCudaError != cudaSuccess) {
			printf("BlaBla3-1\n");
			return;
		}
		m_BottomChunk1 = m_TopChunk2;
		m_TopChunk1 = m_TopChunk2 + m_TopChunk2 - m_BottomChunk2;
	}
	else {
		m_LastCudaError = cudaMemset(m_hHistograms[1], 0, m_NoHistoChunks * m_HistoCellSize);
		if (m_LastCudaError != cudaSuccess) {
			printf("BlaBla3-2\n");
			return;
		}
		m_BottomChunk2 = m_TopChunk1;
		m_TopChunk2 = m_TopChunk1 + m_TopChunk1 - m_BottomChunk1;
	}
}