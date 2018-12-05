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

int main(int argc, char* argv[])
{
    double sigma = 1.0;
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
    printf("Texton1 : %s\n", qPrintable(t3.toString()));

	QString textonDBPath;
	if (argc > 1) {
		textonDBPath = QString(argv[1]);
	}

	FilterBank filterBank;
    TextonGenerator tg(filterBank);
	if (argc > 1)
		tg.setDataPath(textonDBPath);
    tg.execute();

    return 0;
}
