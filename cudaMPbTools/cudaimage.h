#ifndef _CUDA_Image_
#define _CUDA_Image_

class CudaImage {
public:
	CudaImage(unsigned char* image_data, int image_width, int image_height);
	~CudaImage();

	bool wasSuccessfullyCreated() {
		return m_LastCudaError == cudaSuccess;
	}

private:
	unsigned char* m_dSourceImage = nullptr; //image on the GPU
	double* m_GradientImages = nullptr; //the result images

	int m_Width = 0;
	int m_Height = 0;

	cudaError_t m_LastCudaError;

	int m_ArcNo = 4;
	bool m_FullyInitialized = false;
};


#endif

