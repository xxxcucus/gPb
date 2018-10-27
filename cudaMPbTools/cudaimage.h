#ifndef _CUDA_Image_
#define _CUDA_Image_

class CudaImage {
public:
	CudaImage(unsigned char* image_data, int image_width, int image_height, int scale);
	~CudaImage();

	bool wasSuccessfullyCreated() {
		return m_LastCudaError == cudaSuccess;
	}

    /**
     * @brief initializeHistoRange - creates empty histograms for rows of image
     * between start and stop.
     * @param start
     * @param stop
     */
	//TODO: to implement
    bool initializeHistoRange(int start, int stop);	

private:
	unsigned char* m_dSourceImage = nullptr; //image on the GPU
	double* m_dGradientImages = nullptr; //the result images
	unsigned char** m_dHistograms = nullptr;

	int m_Width = 0;
	int m_Height = 0;
	int m_Scale = 0;

	cudaError_t m_LastCudaError;

	int m_ArcNo = 4;
	bool m_FullyInitialized = false;
};


#endif

