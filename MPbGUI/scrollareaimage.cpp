#include "scrollareaimage.h"

ScrollAreaImage::ScrollAreaImage(QWidget* parent) : QScrollArea(parent) {
	setBackgroundRole(QPalette::Dark);
}

void ScrollAreaImage::setImage(cv::Mat image, bool gray) {
	if (m_ImageLabel)
		delete m_ImageLabel;
	
	m_ImageLabel = new QLabel();

	QImage qImg;
	if (!gray) {
		qImg = QImage(reinterpret_cast<uchar*>(image.data), image.cols, image.rows, static_cast<int>(image.step), QImage::Format_RGB888);
	}
	else {
		qImg = QImage(reinterpret_cast<uchar*>(image.data), image.cols, image.rows, static_cast<int>(image.step), QImage::Format_Grayscale8);
	}
	QPixmap pixImg = QPixmap::fromImage(qImg);
	m_ImageLabel->setPixmap(pixImg);

	setWidget(m_ImageLabel);
}