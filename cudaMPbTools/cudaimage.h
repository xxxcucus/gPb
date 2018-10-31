#ifndef _CUDA_Image_
#define _CUDA_Image_

#include "discinversemasks.h"

class CudaImage {
public:
	CudaImage(unsigned char* image_data, int image_width, int image_height, int scale);
	~CudaImage();

	bool wasSuccessfullyCreated() {
		return m_FullyInitialized;
	}

private:
	
    /**
    * Allocates image on the GPU, pads it with zeros, and copies from host to gpu the image data. 
    */
	bool copyImageToGPU();
    /**
    * Creates 2D histogram array in the GPU.
    */
	bool create2DHistoArray();
    /**
    * Allocates memory for the gradient images.
    */
	bool createGradientImages();

    /**
     * Creates empty histograms on the GPU
     * for rows of image between start and stop.
     * @param start
     * @param stop
     */
    bool initializeHistoRange(int start, int stop);	
    /**
     * Copies m_Masks->getHalfDiscInfluencePoints()
     * to the GPU
     */
    bool initializeInfluencePoints();


	void addToHistoArray(int val, int i, int j);

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

	//the half disc masks around a center point
    DiscInverseMasks* m_Masks = nullptr;
};


#endif

