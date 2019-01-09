#include "mpbguicentralwidget.h"
#include <QVBoxLayout>
#include <QFileDialog>

MPbGuiCentralWidget::MPbGuiCentralWidget(QWidget* parent) : QTabWidget(parent) {
	m_ScrollAreaImage = new ScrollAreaImage();
	m_ScrollAreaEdges = new ScrollAreaImage();

	addTab(m_ScrollAreaImage, "Image");
	addTab(m_ScrollAreaEdges, "Edges");
}

void MPbGuiCentralWidget::loadImage() {
	QFileDialog dialog(this);
	dialog.setFileMode(QFileDialog::AnyFile);
	dialog.setNameFilter(tr("Images (*.png *.xpm *.jpg)"));
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

	m_ScrollAreaImage->setImage(m_Image);

	update();
}


