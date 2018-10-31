#include "cudaimage.h"

CudaImage::CudaImage(unsigned char* image_data, int image_width, int image_height, int scale) : 
	m_Width(image_width), m_Height(image_height), m_Scale(scale)
{
	//copy image to the device memory and pad with zeros
	//TODO: for start pad with zeros, but later as in the CPU method
	//another solution is to pad the image before it is given to this class
	m_LastCudaError = cudaMalloc(&m_dSourceImage, (image_width + 2 * m_Scale) * (image_height + 2 * m_Scale));

	if (m_LastCudaError != cudaSuccess)
		return;

	//set all pixels in the image to zero 
	m_LastCudaError = cudaMemset(m_dSourceImage, 0, (image_width + 2 * m_Scale) * (image_height + 2 * m_Scale));

	if (m_LastCudaError != cudaSuccess)
		return;

	//copy from the host image with padding
	int count = 0;
	while (count < image_height && m_LastCudaError == cudaSuccess)
	{
		m_LastCudaError = cudaMemcpy(image_data + count * image_width, m_dSourceImage + m_Scale * (image_width + 2 * m_Scale) + count * (image_width + 2 * m_Scale) + m_Scale, image_width, cudaMemcpyHostToDevice);
		count++;
	}

	if (m_LastCudaError != cudaSuccess)
		return;	

	//allocate the device memory for the gradient images
	m_LastCudaError = cudaMalloc(&m_dGradientImages, m_ArcNo * image_width * image_height * sizeof(double));

	if (m_LastCudaError != cudaSuccess) {
		m_dGradientImages = nullptr;
		return;
	}

	//set all pixels in the gradient images to 0
	m_LastCudaError = cudaMemset(m_dGradientImages, 0, m_ArcNo * image_width * image_height * sizeof(double));

	if (m_LastCudaError != cudaSuccess)
		return;

	//preparing histograms
	m_LastCudaError = cudaMalloc(&m_dHistograms, (m_Height + 2 * m_Scale) * sizeof(unsigned char*));
	for (int i = 0; i < m_Height + 2 * m_Scale; ++i)
		m_dHistograms[i] = nullptr;

	m_FullyInitialized = true;
}

bool CudaImage::initializeHistoRange(int start, int stop)
{ 
    for (int i = start; i < stop; ++i) {
		m_LastCudaError = cudaMalloc(&m_dHistograms[i], 256 * 2 * m_ArcNo * (m_Width + 2 * m_Scale) * sizeof(unsigned char));
		
		if (m_LastCudaError != cudaSuccess)
			return false;

		//set all pixels in the gradient images to 0
		m_LastCudaError = cudaMemset(m_dHistograms[i], 0, 256 * 2 * m_ArcNo * (m_Width + 2 * m_Scale) * sizeof(unsigned char));

		if (m_LastCudaError != cudaSuccess)
			return false;
		}

	return true;
}

void CudaImage::addToHistoMaps(int val, int i, int j)
{
	
}

CudaImage::~CudaImage()
{
	cudaFree(m_dSourceImage);
	cudaFree(m_dGradientImages);
	
	for (int i = 0; i < m_Height + 2 * m_Scale; ++i)
		if (m_dHistograms[i] != nullptr)
			cudaFree(m_dHistograms[i]);

	cudaFree(m_dHistograms);
}