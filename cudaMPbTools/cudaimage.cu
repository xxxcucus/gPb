#include "cudaimage.h"
#include "cvector.h"
#include <cstdlib>

__global__ void calculateGradients(int row, double* dGradientImages, unsigned int** dHistograms, int image_width, int image_height, int scale, int arcno) {
	if (row < 2 * scale)
		return;

	int index = threadIdx.x;
	int stride = blockDim.x;

	for (int j = scale + index; j < image_width + scale; j += stride) {
		unsigned int* vHist = dHistograms[row - scale] +  j * 2 * arcno * 256;

		for (int i = 0; i < arcno; ++i) {
			unsigned int* histo1 = vHist + i * 256;
			unsigned int* histo2 = vHist + (i + arcno) * 256;
			//printf("Chi square for:\n");
			double val = 0.0;
			for (int k = 0; k < 256; ++k) {
				if (histo1[k] != 0 || histo2[k] != 0) {
					double diff = double(int(histo1[k] - histo2[k]));
					double sum = double(histo1[k] + histo2[k]);
					val = val + diff * diff / sum;
					//printf("[%d - %d = %f]", histo1[k], histo2[k], val);
				}

			}
			double grad = val;
			//printf("Grad[%d, %d, %d]=%f\n", row, j, i, grad);
			*(dGradientImages + i * image_width * image_height + (row - 2 * scale) * image_width + (j - scale)) = grad;
		}
	}
}

__global__ void calcHisto(int row, unsigned char* dSourceImage, struct CVector* dHalfDiscInfluencePoints, int totalHalfInfluencePoints, unsigned int** dHistograms, int image_width, int image_height, int scale, int arcno)
{
	int index = threadIdx.x;
	int stride = blockDim.x;

	int i = row;

	for (int j = index; j < image_width + 2 * scale; j += stride) {
		//qDebug() << "BlaBla1 " << j;
		unsigned char val = dSourceImage[i * (image_width + 2 * scale) + j];
		//printf("Index %d Val %d \n", j, int(val));
		//with the point (i,j) with value val, update all histograms which contain this data point
		addToHistoArray(dHalfDiscInfluencePoints, totalHalfInfluencePoints, dHistograms, image_width, image_height, scale, arcno, val, i, j);
	}
}

__device__ void addToHistoArray(struct CVector* dHalfDiscInfluencePoints, int totalHalfInfluencePoints, unsigned int** dHistograms, int image_width, int image_height, int scale, int arcno, int val, int i, int j)
{
	for (int k = 0; k < totalHalfInfluencePoints; ++k) {
		struct CVector n = dHalfDiscInfluencePoints[k];
		if ((n.m_Data[0] + i) < 0 || (n.m_Data[0] + i) >= image_height + 2 * scale)
			continue;
		if ((n.m_Data[1] + j) < 0 || (n.m_Data[1] + j) >= image_width + 2 * scale)
			continue;

		//qDebug() << "Compute at " << i << " with " << (n[0] + i) << " size " << vMaps[n[0] + i].size();
		/*if (int(vMaps[n[0] + i].size()) != m_SingleChannelImage.cols + 2 * m_Scale) {
			qDebug() << "exiting ..";
			exit(1);
		}*/

		unsigned int* vHist = dHistograms[n.m_Data[0] + i] + (n.m_Data[1] + j) * 2 * arcno * 256;
		for (unsigned int l = 2; l < n.m_Size; ++l) {
			if (n.m_Data[l] > 2 * arcno)
				continue;
			//qDebug() << "Insert into histo " << n[k] << " val " << val << " vHist size " << vHist.size();
			//TODO: use atomic operation
			atomicInc(vHist + n.m_Data[l] * 256 + val, 4 * scale * scale);
		}
	}
}

CudaImage::CudaImage(unsigned char* image_data, int image_width, int image_height, int scale) :
	m_Width(image_width), m_Height(image_height), m_Scale(scale)
{
	if (!createGradientImages()) {
		printf("Error in constructor createGradientImages\n");
		return;
	}

	if (!copyImageToGPU(image_data)) {
		printf("Error in constructor copyImageToGPU\n");
		return;
	}

	if (!create2DHistoArray()) {
		printf("Error in constructor create2DHistoArray\n");
		return;
	}

	if (!initializeInfluencePoints()) {
		printf("Error in constructor initializeInfluencePoints\n");
		return;
	}

	m_FullyInitialized = true;
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

	m_hGradientImages = (double*)malloc(m_ArcNo * m_Width * m_Height * sizeof(double));

	return true;
}

bool CudaImage::initializeHistoRange(int start, int stop)
{
	for (int i = start; i < stop; ++i) {
		m_LastCudaError = cudaMalloc((void**)&m_hHistograms[i], 256 * 2 * m_ArcNo * (m_Width + 2 * m_Scale) * sizeof(unsigned int));
		if (m_LastCudaError != cudaSuccess) {
			return false;
		}

		//set all pixels in the gradient images to 0
		//m_LastCudaError = cudaMemset(m_dHistograms[i], 0, 256 * 2 * m_ArcNo * (m_Width + 2 * m_Scale) * sizeof(unsigned int));
		//if (m_LastCudaError != cudaSuccess)
		//	return false;
		//}

		cudaMemcpy(m_dHistograms, m_hHistograms, stop * sizeof(unsigned int*), cudaMemcpyHostToDevice);
		if (m_LastCudaError != cudaSuccess)
			return false;
	}

	return true;
}

void CudaImage::deleteFromHistoMaps(int index) {
	//add a new row in the vMaps
	if (index + m_Scale + 1 < m_Height + 2 * m_Scale) {
		initializeHistoRange(index + m_Scale + 1, index + m_Scale + 2);
		//qDebug() << "Initialize " << index + m_Scale + 1;
	}
	//delete row which was already analyzed from vMaps
	if (index >= m_Scale + 1) {
		//qDebug() << "Delete " << index - m_Scale;
		cudaFree(m_hHistograms[index - m_Scale - 1]);
		m_hHistograms[index - m_Scale - 1] = nullptr;
	}
}

bool CudaImage::create2DHistoArray()
{
	//preparing histograms
	m_LastCudaError = cudaMalloc((void**)&m_dHistograms, (m_Height + 2 * m_Scale) * sizeof(unsigned int*));

	if (m_LastCudaError != cudaSuccess)
		return false;
	
	//set all histograms to nullptr
	//m_LastCudaError = cudaMemset(m_dHistograms, 0, (m_Height + 2 * m_Scale) * sizeof(unsigned int*));

	if (m_LastCudaError != cudaSuccess)
		return false;

	m_hHistograms = (unsigned int**)malloc((m_Height + 2 * m_Scale) * sizeof(unsigned int*));

	return true;
}

//TODO: release of host memory
CudaImage::~CudaImage()
{
	cudaFree(m_dSourceImage);
	cudaFree(m_dGradientImages);
	
	for (int i = 0; i < m_Height + 2 * m_Scale; ++i) {
		if (m_hHistograms[i])
			cudaFree(m_hHistograms[i]);
	}

	free(m_hHistograms);
	cudaFree(m_dHistograms);

	for (int i = 0; i < m_TotalHalfInfluencePoints; ++i) {
		cudaFree(m_hHalfDiscInfluencePoints[i].m_Data); 
	}

	free(m_hHalfDiscInfluencePoints);
	cudaFree(m_dHalfDiscInfluencePoints);
}

bool CudaImage::copyImageToGPU(unsigned char* image_data)
{
	//copy image to the device memory and pad with zeros
	//TODO: for start pad with zeros, but later as in the CPU method
	//another solution is to pad the image before it is given to this class
	m_LastCudaError = cudaMalloc((void**)&m_dSourceImage, (m_Width + 2 * m_Scale) * (m_Height + 2 * m_Scale));

	if (m_LastCudaError != cudaSuccess) {
		printf("Error copyImageToGPU cudaMalloc %s, %d %d %d\n", cudaGetErrorString(m_LastCudaError), m_Width, m_Height, m_Scale);
		//return false;
	}

	//set all pixels in the image to zero 
	m_LastCudaError = cudaMemset(m_dSourceImage, 0, (m_Width + 2 * m_Scale) * (m_Height + 2 * m_Scale));

	if (m_LastCudaError != cudaSuccess) {
		printf("Error copyImageToGPU cudaMemset\n");
		return false;
	}

	//copy from the host image with padding
	int count = 0;
	while (count < m_Height && m_LastCudaError == cudaSuccess)
	{
		m_LastCudaError = cudaMemcpy(m_dSourceImage + m_Scale * (m_Width + 2 * m_Scale) + count * (m_Width + 2 * m_Scale) + m_Scale, image_data + count * m_Width, m_Width, cudaMemcpyHostToDevice);
		if (m_LastCudaError != cudaSuccess) {
			printf("Error copyImageToGPU cudaMemcpy %d\n", count);
			return false;
		}
		count++;
	}
	
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

	m_hHalfDiscInfluencePoints = (CVector*)malloc(m_TotalHalfInfluencePoints * sizeof(CVector));
	for (int i = 0; i < m_TotalHalfInfluencePoints; ++i) {
		m_hHalfDiscInfluencePoints[i].m_Size = int(neighb[i].size());
		m_LastCudaError = cudaMalloc(&m_hHalfDiscInfluencePoints[i].m_Data, neighb[i].size() * sizeof(int));
		if (m_LastCudaError != cudaSuccess)
			return false;
		int* values = (int*)malloc(neighb[i].size() * sizeof(int));
		
		for (int j = 0; j < neighb[i].size(); ++j)
			values[j] = neighb[i][j];
		m_LastCudaError = cudaMemcpy(m_hHalfDiscInfluencePoints[i].m_Data, values, neighb[i].size() * sizeof(int), cudaMemcpyHostToDevice);
		if (m_LastCudaError != cudaSuccess)
			return false;
		free(values);
	}

//TODO: release memory in case of failure

	//preparing histograms
	m_LastCudaError = cudaMalloc(&m_dHalfDiscInfluencePoints, m_TotalHalfInfluencePoints * sizeof(CVector));
	if (m_LastCudaError != cudaSuccess)
		return false;

	cudaMemcpy(m_dHalfDiscInfluencePoints, m_hHalfDiscInfluencePoints, m_TotalHalfInfluencePoints * sizeof(CVector), cudaMemcpyHostToDevice);
	if (m_LastCudaError != cudaSuccess)
		return false;


	//tests if data was correctly copied to device
	/*
	for (int i = 0; i < m_TotalHalfInfluencePoints; ++i) {
		int* values = (int*)malloc(m_hHalfDiscInfluencePoints[i].m_Size * sizeof(int));
		cudaMemcpy(values, m_hHalfDiscInfluencePoints[i].m_Data, m_hHalfDiscInfluencePoints[i].m_Size * sizeof(int), cudaMemcpyDeviceToHost);
		printf("Size device: %d\n", m_hHalfDiscInfluencePoints[i].m_Size);
		for (int j = 0; j < m_hHalfDiscInfluencePoints[i].m_Size; ++j)
			printf("%d ", values[j]);
		printf("\n");
		printf("Size host: %d\n", neighb[i].size());
		for (int j = 0; j < neighb[i].size(); ++j)
			printf("%d ", neighb[i][j]);
		printf("\n");
	}*/

	return true;	
}

void CudaImage::execute() {
	//printf("BlaBla 10\n");
	initializeHistoRange(0, m_Scale + 1);
	//printf("BlaBla 11\n");

	for (int i = 0; i < m_Height + 2 * m_Scale; ++i) {
		//printf("%d - BlaBla 1\n", i);
		calcHisto<<<1, 1024>>>(i, m_dSourceImage, m_dHalfDiscInfluencePoints, m_TotalHalfInfluencePoints, m_dHistograms, m_Width, m_Height, m_Scale, m_ArcNo);
		cudaDeviceSynchronize();
		//printf("%d - BlaBla 2\n", i);
		calculateGradients<<<1, 1024>>>(i, m_dGradientImages, m_dHistograms, m_Width, m_Height, m_Scale, m_ArcNo);
		cudaDeviceSynchronize();
		//printf("%d - BlaBla 3\n", i);
		deleteFromHistoMaps(i);
		//printf("%d - BlaBla 4\n", i);
	}

	m_LastCudaError = cudaMemcpy(m_hGradientImages, m_dGradientImages, m_ArcNo * m_Width * m_Height * sizeof(double), cudaMemcpyDeviceToHost);
}