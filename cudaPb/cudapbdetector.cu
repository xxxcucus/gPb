#include "cudapbdetector.h"
#include "cvector.h"
#include <cstdlib>


__global__ void calculateGradients(int row_start, int row_count, double* dGradientImages, \
	unsigned int** dHistograms, int bottomChunk1, int bottomChunk2, int topChunk1, int topChunk2,\
	int image_width, int image_height, int scale, int arcno) {
	int row = row_start + blockIdx.x;
	int index = threadIdx.x;
	int stride = blockDim.x;
	
	if (row < 2 * scale || row >= image_height + 2 * scale || row >= row_start + row_count)
		return;

	for (int j = scale + index; j < image_width + scale; j += stride) {
		//unsigned int* vHist = dHistograms[row - scale] +  j * 2 * arcno * 256;
		unsigned int* vHist = getHistoPointer(row - scale, j, dHistograms, bottomChunk1, bottomChunk2, topChunk1, topChunk2, image_width, scale, arcno);
		
		//todo: error handling
		if (vHist == nullptr)
			return;

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

__global__ void calcHisto(int row_start, int row_count, unsigned char* dSourceImage,\
	int* dHalfDiscInfluencePoints, int totalHalfInfluencePoints, \
	unsigned int** dHistograms, int bottomChunk1, int bottomChunk2, int topChunk1, int topChunk2,\
	int image_width, int image_height, int scale, int arcno)
{
	int index = threadIdx.x;
	int stride = blockDim.x;
	int i = row_start + blockIdx.x;

	if (blockIdx.x > row_count)
		return;

	for (int j = index; j < image_width + 2 * scale; j += stride) {
		//qDebug() << "BlaBla1 " << j;
		unsigned char val = dSourceImage[i * (image_width + 2 * scale) + j];
		//printf("Row %d Index %d Val %d \n", i, j, int(val));
		//with the point (i,j) with value val, update all histograms which contain this data point
		addToHistoArray(dHalfDiscInfluencePoints, totalHalfInfluencePoints, dHistograms, bottomChunk1, bottomChunk2, topChunk1, topChunk2, image_width, image_height, scale, arcno, val, i, j);
	}
}

__device__ void addToHistoArray(int* dHalfDiscInfluencePoints, int totalHalfInfluencePoints,\
	unsigned int** dHistograms, int bottomChunk1, int bottomChunk2, int topChunk1, int topChunk2, \
	int image_width, int image_height, int scale, int arcno, int val, int i, int j)
{
	int data_size = 1;
	for (int k = 0; k < totalHalfInfluencePoints; k += data_size + 1) {
		data_size = dHalfDiscInfluencePoints[k];
		int* data = dHalfDiscInfluencePoints + k + 1; 
		int row = data[0] + i;
		int col = data[1] + j;
		if (row < 0 || row >= image_height + 2 * scale)
			continue;
		if (col < 0 || col >= image_width + 2 * scale)
			continue;

		//printf("Compute at  %d %d\n", n.m_Data[0] + i, n.m_Data[1] + j);
		/*if (int(vMaps[n[0] + i].size()) != m_SingleChannelImage.cols + 2 * m_Scale) {
			qDebug() << "exiting ..";
			exit(1);
		}*/

		//unsigned int* vHist = dHistograms[n.m_Data[0] + i] + (n.m_Data[1] + j) * 2 * arcno * 256;
		unsigned int* vHist = getHistoPointer(row, col, dHistograms, bottomChunk1, bottomChunk2, topChunk1, topChunk2, image_width, scale, arcno);
		//todo: error handling
		if (vHist == nullptr)
			continue;
		
		for (unsigned int l = 2; l < dHalfDiscInfluencePoints[k]; ++l) {
			int idx = data[l];
			if (idx >= 2 * arcno || val < 0 || val >= 256)
				continue;
			//qDebug() << "Insert into histo " << n[k] << " val " << val << " vHist size " << vHist.size();
			//TODO: use atomic operation
			atomicInc(vHist + idx * 256 + val, 4 * scale * scale);
		}
	}
}

__device__ unsigned int* getHistoPointer(int row, int col, \
	unsigned int** dHistograms, int bottomChunk1, int bottomChunk2, int topChunk1, int topChunk2,\
	int width, int scale, int arcno) {
	
	int bottomChunk = bottomChunk1;
	if (bottomChunk2 < bottomChunk1)
		bottomChunk = bottomChunk2;
	int topChunk = topChunk1;
	if (topChunk2 > topChunk1)
		topChunk = topChunk2;

	//printf("Bottom1 %d - Top1 %d\n Bottom2 %d - Top2 %d \n", bottomChunk1, topChunk1, bottomChunk2, topChunk2);
	//printf("Bottom %d - Top %d \n", bottomChunk, topChunk);
	
	if (row < bottomChunk || row >= topChunk)
		return nullptr;

	if (col < 0 || col >= width + 2 * scale)
		return nullptr;

	if (topChunk1 - bottomChunk1 != topChunk2 - bottomChunk2) {
		printf("Error difference 1\n");
		return nullptr;
	}

	if (topChunk - bottomChunk != 2 * (topChunk1 - bottomChunk1)) {
		printf("Error difference 2\n");
		return nullptr;
	}

	unsigned int* rowp;
	int middleChunk = (bottomChunk + topChunk) / 2;
	//printf("Row %d Middlechunk %d\n", row, middleChunk);
	if (row < middleChunk) {
		if ((row - bottomChunk < 0) || (row - bottomChunk >= topChunk1 - bottomChunk1)) {
			printf("Error bottomChunk!!!!\n");
			return nullptr;
		}
		long int d = 256 * 2 * long int(arcno) * long int(width + 2 * scale) * long int(row - bottomChunk);
		//printf("Offset %ld  Arcno %d Width %d Scale %d Factor %d\n", d, arcno, width, scale, row - bottomChunk);
		if (bottomChunk1 < bottomChunk2)
			rowp = dHistograms[0] + d;
		else
			rowp = dHistograms[1] + d;
	}
	else {
		if ((row - middleChunk < 0) || (row - middleChunk >= topChunk1 - bottomChunk1 - 1)) {
			printf("Error middleChunk!!!!\n");
			return nullptr;
		}
		long int d = 256 * 2 * long int(arcno) * long int(width + 2 * scale) * long int(row - middleChunk);

		if (bottomChunk1 < bottomChunk2)
			rowp = dHistograms[1] + d;
		else
			rowp = dHistograms[0] + d;
	}

	return rowp + 256 * 2 * arcno * col;
}


CudaPbDetector::CudaPbDetector(unsigned char* image_data, int image_width, int image_height, int scale) :
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

	/*if (!create2DHistoArray()) {
		printf("Error in constructor create2DHistoArray\n");
		return;
	}*/

	if (!initializeInfluencePoints()) {
		printf("Error in constructor initializeInfluencePoints\n");
		return;
	}

	printf("Constructing histo allocator\n");
	m_HistoAllocator = new HistoAllocator(m_Width, m_Height, m_ArcNo, m_Scale, m_Step);
	if (m_HistoAllocator->wasError())
		return;

	readDeviceProperties();

	m_FullyInitialized = true;
}

bool CudaPbDetector::createGradientImages()
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

bool CudaPbDetector::initializeHistoRange(int start, int stop)
{
	//printf("InitializeHistoRange %d-%d\n", start, stop);
	int topChunk = std::max(m_HistoAllocator->m_TopChunk1, m_HistoAllocator->m_TopChunk2);
	int bottomChunk = std::min(m_HistoAllocator->m_BottomChunk1, m_HistoAllocator->m_BottomChunk2);
	
	if (stop < topChunk && start >= bottomChunk) {
		return true;
	}

	int middleChunk = (topChunk + bottomChunk) / 2;

	if (start < middleChunk) {
		printf("Error start below middleChunk\n");
		return false;
	}

	//TODO: when stop > m_HistoAllocator->m_TopChunk2 we must be sure that we do not need the bottom chunk anymore!!! - be carefull streaming

	m_HistoAllocator->setNewTopChunk();

	if (m_HistoAllocator->wasError()) {
		printf("Error when allocating new chunk - %s\n", cudaGetErrorString(m_HistoAllocator->getError()));
		return false;
	}
	
	return true;
}

bool CudaPbDetector::updateHistoBuffer(int step, int index) {

	if (index + m_Scale + step + 1 < m_Height + 2 * m_Scale) {
		if (!initializeHistoRange(index + m_Scale + step + 1, index + m_Scale + step + 2))
			return false;
	}

	return true;
}

bool CudaPbDetector::create2DHistoArray()
{
	//preparing histograms
	m_LastCudaError = cudaMalloc((void**)&m_dHistograms, (m_Height + 2 * m_Scale) * sizeof(unsigned int*));

	if (m_LastCudaError != cudaSuccess)
		return false;

	m_hHistograms = (unsigned int**)malloc((m_Height + 2 * m_Scale) * sizeof(unsigned int*));

	return true;
}

//TODO: to check this
CudaPbDetector::~CudaPbDetector()
{
	cudaFree(m_dSourceImage);
	cudaFree(m_dGradientImages);

	delete m_HistoAllocator;

	free(m_hHalfDiscInfluencePoints);
	cudaFree(m_dHalfDiscInfluencePoints);
}

bool CudaPbDetector::copyImageToGPU(unsigned char* image_data)
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
/*bool CudaPbDetector::initializeInfluencePoints() {
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

	return true;	
}*/

bool CudaPbDetector::initializeInfluencePoints() {
	m_Masks = new DiscInverseMasks(m_Scale);
	std::vector<std::vector<int>> neighb = m_Masks->getHalfDiscInfluencePoints();

	//save the vectors inside a single linear vector
	std::vector<int> linear_neighb;
	for (auto nb : neighb) {
		linear_neighb.push_back(int(nb.size()));
		for (auto ne : nb)
			linear_neighb.push_back(ne);
	}

	m_TotalHalfInfluencePoints = int(linear_neighb.size());
	//printf("Total influence points %d\n", m_TotalHalfInfluencePoints);

	m_hHalfDiscInfluencePoints = (int*)malloc(m_TotalHalfInfluencePoints * sizeof(int));
	for (unsigned int i = 0; i < linear_neighb.size(); ++i) {
		//printf("[%d] = %d\n", i, linear_neighb[i]);
		m_hHalfDiscInfluencePoints[i] = linear_neighb[i];
	}

	//preparing histograms
	m_LastCudaError = cudaMalloc(&m_dHalfDiscInfluencePoints, m_TotalHalfInfluencePoints * sizeof(int));
	if (m_LastCudaError != cudaSuccess)
		return false;

	cudaMemcpy(m_dHalfDiscInfluencePoints, m_hHalfDiscInfluencePoints, m_TotalHalfInfluencePoints * sizeof(int), cudaMemcpyHostToDevice);
	if (m_LastCudaError != cudaSuccess)
		return false;

	return true;
}

bool CudaPbDetector::executeChunk() {

	if (!initializeHistoRange(0, m_Scale + m_Step + 1))
		return false;
	int noSteps = (m_Height + 2 * m_Scale + m_Step - 1) / m_Step;

	for (int i = 0; i < noSteps; ++i) {
		int row_start = m_Step * i;
		int row_count = std::min(m_Step, m_Height + 2 * m_Scale - row_start);
		//printf("Row_start: %d Row_count %d Scale %d BottomChunk1 %d TopChunk1 %d BottomChunk2 %d TopChunk2 %d \n", row_start, row_count, m_Scale, m_HistoAllocator->m_BottomChunk1, m_HistoAllocator->m_TopChunk1, m_HistoAllocator->m_BottomChunk2, m_HistoAllocator->m_TopChunk2);
		calcHisto<<<row_count, m_NoThreads>>>(row_start, row_count, m_dSourceImage, m_dHalfDiscInfluencePoints, m_TotalHalfInfluencePoints, \
			m_HistoAllocator->m_dHistograms, m_HistoAllocator->m_BottomChunk1, m_HistoAllocator->m_BottomChunk2, m_HistoAllocator->m_TopChunk1, m_HistoAllocator->m_TopChunk2, \
			m_Width, m_Height, m_Scale, m_ArcNo);
		cudaDeviceSynchronize();
		m_LastCudaError = cudaGetLastError();
		if (m_LastCudaError != cudaSuccess) {
			printf("Error execution 1\n");
			return false;
		}
		calculateGradients << <row_count, m_NoThreads >> > (row_start, row_count, m_dGradientImages, \
			m_HistoAllocator->m_dHistograms, m_HistoAllocator->m_BottomChunk1, m_HistoAllocator->m_BottomChunk2, m_HistoAllocator->m_TopChunk1, m_HistoAllocator->m_TopChunk2, \
			m_Width, m_Height, m_Scale, m_ArcNo);
		cudaDeviceSynchronize();
		m_LastCudaError = cudaGetLastError();
		if (m_LastCudaError != cudaSuccess) {
			printf("Error execution 2\n");
			return false;
		}
		for (int k = row_start; k < row_count + row_start; ++k) {
			if (!updateHistoBuffer(m_Step, k))
				return false;
		}	
	}

	m_LastCudaError = cudaMemcpy(m_hGradientImages, m_dGradientImages, m_ArcNo * m_Width * m_Height * sizeof(double), cudaMemcpyDeviceToHost);
	return m_LastCudaError == cudaSuccess;	
}


void CudaPbDetector::readDeviceProperties() {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	m_SharedMemoryPerBlock = deviceProp.sharedMemPerBlock;
}