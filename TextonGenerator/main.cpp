#include <QDebug>
#include <QString>
#include "opencv2/opencv.hpp"
#include "gaussianfirstderivativefilter.h"
#include "gaussiansecondderivativefilter.h"
#include "gaussianfilter.h"
#include "textonkernel.h"
#include "textongenerator.h"
#include "texton.h"
#include "filterbank.h"
#include "argumentlist.h"

int main(int argc, char* argv[])
{
/*    double sigma = 1.0;
    for (int k = 0; k < 0; ++k) {
        for (int i = 0; i < 8; ++i) {
            printf("First derivative filter for %f degrees %f sigma\n", double(i) * 22.5, sigma);
            GaussianFirstDerivativeFilter gdf(15, double(i) * 22.5, sigma);
            gdf.init();
        }
        sigma *= sqrt(2.0);
    }

    sigma = 3.0;
    for (int k = 0; k < 0; ++k) {
        for (int i = 0; i < 8; ++i) {
            printf("Second derivative filter for %f degrees %f sigma\n", double(i) * 22.5, sigma);
            GaussianSecondDerivativeFilter gdf(15, double(i) * 22.5, sigma);
            gdf.init();
        }
        sigma *= sqrt(2.0);
    }

    sigma = 1.0;
    for (int k = 0; k < 0; ++k) {
        printf("LOG Filter for %f sigma\n", sigma);
        GaussianSecondDerivativeFilter gdf1(15, 45.0, sigma, 1.0, 1.0);
        gdf1.init();
        printf("LOG Filter for %f sigma\n", 3 * sigma);
        GaussianSecondDerivativeFilter gdf2(15, 45.0, 3 * sigma, 1.0, 1.0);
        gdf2.init();
        sigma *= sqrt(2.0);
    }

    sigma = 1.0;
    for (int k = 0; k < 0; ++k) {
        printf("Gaussian Filter"
               " for %f sigma\n", sigma);
        GaussianFilter gdf(15, sigma);
        gdf.init();
        sigma *= sqrt(2.0);
    }

    Texton t1;
    Texton t2;
    for (int i = 0; i < 17; ++i) {
        t1.setValueAtIdx(i, i);
        t2.setValueAtIdx(1, i);
    }


    Texton t3 = t1 + t2;
    t3 = t3 / 2.0;
    printf("Texton1 : %s\n", qPrintable(t3.toString()));*/

	ArgumentList al(argc, argv);

	if (al.getSwitch("--help")) {
		printf("TextonGenerator --help - shows this help\n");
		printf("TextonGenerator --textonFile FileName --textureFolder FolderName \n");
		printf("\t --textonFile defines the file in which the textons will be saved \n");
		printf("\t --textureFolder defines the folder where the images for texture quantization will be saved\n");
		return 0;
	}

	FilterBank filterBank;

	QString textureFolderPath = al.getSwitchArg("--textureFolder", QString());

	TextonGenerator tg(filterBank, textureFolderPath);

	QString textonFilePath = al.getSwitchArg("--textonFile", QString());
	if (!textonFilePath.isEmpty()) {
		tg.setTargetPath(textonFilePath);
	}
		
    tg.execute();

    return 0;
}
