#include "cudaimage.h"
#include "cvector.h"
#include <cstdlib>

__device__ void CudaImage::addToHistoArray(int val, int i, int j)
{
	for (int k = 0; k < m_TotalHalfInfluencePoints; ++k) {
		struct CVector n = m_dHalfDiscInfluencePoints[k];
		if ((n.m_Data[0] + i) < 0 || (n.m_Data[0] + i) >= m_Height + 2 * m_Scale)
			continue;
		if ((n.m_Data[1] + j) < 0 || (n.m_Data[1] + j) >= m_Width + 2 * m_Scale)
			continue;

		//qDebug() << "Compute at " << i << " with " << (n[0] + i) << " size " << vMaps[n[0] + i].size();
		/*if (int(vMaps[n[0] + i].size()) != m_SingleChannelImage.cols + 2 * m_Scale) {
			qDebug() << "exiting ..";
			exit(1);
		}*/

		unsigned int* vHist = m_dHistograms[n.m_Data[0] + i] + (n.m_Data[1] + j) * 2 * m_ArcNo * 256;
		for (unsigned int l = 2; l < n.m_Size; ++l) {
			if (n.m_Data[l] > 2 * m_ArcNo)
				continue;
			//qDebug() << "Insert into histo " << n[k] << " val " << val << " vHist size " << vHist.size();
			//TODO: use atomic operation
			atomicInc(vHist + n.m_Data[l] * 256 + val, 4 * m_Scale * m_Scale);
		}
	}
}


CudaImage::CudaImage(unsigned char* image_data, int image_width, int image_height, int scale) : 
	m_Width(image_width), m_Height(image_height), m_Scale(scale)
{
	if (!copyImageToGPU(image_data)) 
		return;

	if (!createGradientImages()) 
		return;

	if (!create2DHistoArray())
		return;

	m_FullyInitialized = true;
}

bool CudaImage::initializeHistoRange(int start, int stop)
{ 
    for (int i = start; i < stop; ++i) {
		m_LastCudaError = cudaMalloc(&m_dHistograms[i], 256 * 2 * m_ArcNo * (m_Width + 2 * m_Scale) * sizeof(unsigned int));
		
		if (m_LastCudaError != cudaSuccess)
			return false;

		//set all pixels in the gradient images to 0
		m_LastCudaError = cudaMemset(m_dHistograms[i], 0, 256 * 2 * m_ArcNo * (m_Width + 2 * m_Scale) * sizeof(unsigned int));

		if (m_LastCudaError != cudaSuccess)
			return false;
		}

	return true;
}

bool CudaImage::createGradientImages() 
{
	//allocate the device memory for the gradient images
	m_LastCudaError = cudaMalloc(&m_dGradientImages, m_ArcNo * m_Width * m_Height * sizeof(double));

	if (m_LastCudaError != cudaSuccess) {
		return false;
	}

	//set all pixels in the gradient images to 0
	m_LastCudaError = cudaMemset(m_dGradientImages, 0, m_ArcNo * m_Width * m_Height * sizeof(double));

	if (m_LastCudaError != cudaSuccess)
		return false;	

	return true;
}

bool CudaImage::create2DHistoArray()
{
	//preparing histograms
	m_LastCudaError = cudaMalloc(&m_dHistograms, (m_Height + 2 * m_Scale) * sizeof(unsigned int*));

	if (m_LastCudaError != cudaSuccess)
		return false;
	
	//set all histograms to nullptr
	m_LastCudaError = cudaMemset(m_dHistograms, 0, (m_Height + 2 * m_Scale) * sizeof(unsigned int*));

	if (m_LastCudaError != cudaSuccess)
		return false;

	return true;
}


//TODO: release of host memory
CudaImage::~CudaImage()
{
	cudaFree(m_dSourceImage);
	cudaFree(m_dGradientImages);
	
	for (int i = 0; i < m_Height + 2 * m_Scale; ++i)
		cudaFree(m_dHistograms[i]);

	cudaFree(m_dHistograms);

	for (int i = 0; i < m_TotalHalfInfluencePoints; ++i) {
		free(m_hHalfDiscInfluencePoints[i].m_Data);
		cudaFree(m_dHalfDiscInfluencePoints[i].m_Data); 
	}

	free(m_hHalfDiscInfluencePoints);
	cudaFree(m_dHalfDiscInfluencePoints);

}

bool CudaImage::copyImageToGPU(unsigned char* image_data)
{
	//copy image to the device memory and pad with zeros
	//TODO: for start pad with zeros, but later as in the CPU method
	//another solution is to pad the image before it is given to this class
	m_LastCudaError = cudaMalloc(&m_dSourceImage, (m_Width + 2 * m_Scale) * (m_Height + 2 * m_Scale));

	if (m_LastCudaError != cudaSuccess)
		return false;

	//set all pixels in the image to zero 
	m_LastCudaError = cudaMemset(m_dSourceImage, 0, (m_Width + 2 * m_Scale) * (m_Height + 2 * m_Scale));

	if (m_LastCudaError != cudaSuccess)
		return false;

	//copy from the host image with padding
	int count = 0;
	while (count < m_Height && m_LastCudaError == cudaSuccess)
	{
		m_LastCudaError = cudaMemcpy(m_dSourceImage + m_Scale * (m_Width + 2 * m_Scale) + count * (m_Width + 2 * m_Scale) + m_Scale, image_data + count * m_Width, m_Width, cudaMemcpyHostToDevice);
		count++;
	}

	if (m_LastCudaError != cudaSuccess)
		return false;	
	
	return true;
}

/**
 * Copies m_Masks->getHalfDiscInfluencePoints()
 * to the GPU
 */
bool CudaImage::initializeInfluencePoints() {
	m_Masks = new DiscInverseMasks(m_Scale);
	std::vector<std::vector<int>> neighb = m_Masks->getHalfDiscInfluencePoints();

	m_TotalHalfInfluencePoints = int(neighb.size());

	m_hHalfDiscInfluencePoints = (CVector*)malloc(neighb.size() * sizeof(CVector));
	for (int i = 0; i < m_TotalHalfInfluencePoints; ++i) {
		m_hHalfDiscInfluencePoints[i].m_Size = int(neighb[i].size());
		m_hHalfDiscInfluencePoints[i].m_Data = (int*)malloc(neighb[i].size() * sizeof(int));
		for (int j = 0; j < neighb[i].size(); ++j)
			m_hHalfDiscInfluencePoints[i].m_Data[j] = neighb[i][j];
	}

//TODO: release memory in case of failure

	//preparing histograms
	m_LastCudaError = cudaMalloc(&m_dHalfDiscInfluencePoints, neighb.size() * sizeof(CVector));

	if (m_LastCudaError != cudaSuccess)
		return false;
		
	
	for (int i = 0; i < m_TotalHalfInfluencePoints; ++i) {
		m_LastCudaError = cudaMalloc(&m_dHalfDiscInfluencePoints[i].m_Data, neighb[i].size() * sizeof(int));
		if (m_LastCudaError != cudaSuccess)
			return false;
		m_LastCudaError = cudaMemcpy(m_dHalfDiscInfluencePoints[i].m_Data, m_hHalfDiscInfluencePoints[i].m_Data, neighb[i].size() * sizeof(int), cudaMemcpyHostToDevice);
		if (m_LastCudaError != cudaSuccess)
			return false;
	}

	return true;	
}


void CudaImage::calculateHistograms() {

}