#include "mpbguicentralwidget.h"
#include <QVBoxLayout>
#include <QFileDialog>
#include <chrono>
#include "mPb.h"

MPbGuiCentralWidget::MPbGuiCentralWidget(QWidget* parent) : QTabWidget(parent) {
	m_ScrollAreaImage = new ScrollAreaImage();
	m_ScrollAreaEdges = new ScrollAreaImage();
	m_ScrollAreaMixed = new ScrollAreaImage();

	addTab(m_ScrollAreaImage, "Image");
	addTab(m_ScrollAreaEdges, "Edges");
	addTab(m_ScrollAreaMixed, "Mixed");
}

void MPbGuiCentralWidget::loadImage() {
	QFileDialog dialog(this);
	dialog.setFileMode(QFileDialog::AnyFile);
	dialog.setNameFilter(tr("Images (*.png *.xpm *.jpg *.ppm)"));
	dialog.setViewMode(QFileDialog::Detail);
	dialog.setDirectory(".");

	QStringList fileNames;
	if (dialog.exec())
		fileNames = dialog.selectedFiles();

	if (fileNames.size() != 1)
		return;

	cv::Mat cvImg_col = cv::imread(fileNames[0].toUtf8().constData());
	cv::cvtColor(cvImg_col, cvImg_col, cv::COLOR_BGR2RGB);
	m_Image = cvImg_col.clone();

	m_ScrollAreaImage->setImage(m_Image, false);

	std::map<std::string, std::vector<int>> mapScales;
	std::vector<int> scales = { 3, 5, 7 };
	mapScales["l"] = scales;
	mapScales["a"] = scales;
	mapScales["b"] = scales;
	mapScales["t"] = scales;

	MultiscalePb detector(m_Image, "textons.txt", mapScales);
	auto grad_start = std::chrono::high_resolution_clock::now();
	detector.computeGradients();
	auto grad_stop = std::chrono::high_resolution_clock::now();
	auto grad_duration = std::chrono::duration_cast<std::chrono::milliseconds>(grad_stop - grad_start);
	printf("Multiscale gradients runtime(ms) %d\n", int(grad_duration.count()));
	detector.computeEdges();
	cv::Mat m_EdgesImage = detector.getEdges();
	cv::imwrite("edges_gui.png", m_EdgesImage);

	m_ScrollAreaEdges->setImage(m_EdgesImage, true);

	/*cv::Mat edgesImageRGB;
	cv::cvtColor(m_EdgesImage, edgesImageRGB, cv::COLOR_GRAY2BGR);
	cv::Mat mixedImage;
	cv::add(m_Image, m_EdgesImage, mixedImage);

	m_ScrollAreaMixed->setImage(mixedImage, false);*/

	update();
}


