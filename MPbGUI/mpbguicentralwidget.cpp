#include "mpbguicentralwidget.h"
#include <QVBoxLayout>
#include <QFileDialog>

MPbGuiCentralWidget::MPbGuiCentralWidget(QWidget* parent) : QWidget(parent) {
	QVBoxLayout* vlayout = new QVBoxLayout();
	m_ImageLabel = new QLabel();
	vlayout->addWidget(m_ImageLabel);
	setLayout(vlayout);
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

	QImage qImg = QImage(reinterpret_cast<uchar*>(cvImg_col.data), cvImg_col.cols, cvImg_col.rows, static_cast<int>(cvImg_col.step), QImage::Format_RGB888);
	QPixmap pixImg = QPixmap::fromImage(qImg);
	QPixmap scaledPixmap = pixImg.scaled(600, 400, Qt::KeepAspectRatio);
	m_ImageLabel->setPixmap(scaledPixmap);
	update();
}


