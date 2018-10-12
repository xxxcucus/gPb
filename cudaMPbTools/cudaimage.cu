#include "cudaimage.h"

CudaImage::CudaImage(unsigned char* image_data, int image_width, int image_height) : 
	m_Width(image_width), m_Height(image_height)
{
	m_LastCudaError = cudaMalloc(&m_dSourceImage, image_width * image_height);

	if (m_LastCudaError == cudaSuccess) {
		m_LastCudaError = cudaMemcpy(image_data, m_dSourceImage, image_width * image_height, cudaMemcpyDeviceToHost);
	}
	else {
		m_dSourceImage = nullptr;
		return;
	}

	m_LastCudaError = cudaMalloc(&m_GradientImages, m_ArcNo * image_width * image_height * sizeof(double));

	if (m_LastCudaError != cudaSuccess)
		return;

	m_LastCudaError = cudaMemcpy(m_GradientImages, m_ArcNo * image_width * image_height * sizeof(double));

	if (m_LastCudaError != cudaSuccess)
		return;

	m_FullyInitialized = true;
}


CudaImage::~CudaImage()
{
	cudaFree(m_dSourceImage);
}