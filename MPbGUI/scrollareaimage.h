#ifndef __SCROLLAREAIMAGE__
#define __SCROLLAREAIMAGE__
#include "opencv2/opencv.hpp"
#include <QLabel>
#include <QScrollArea>
#include <QWidget>

class ScrollAreaImage : public QScrollArea {

public:
	ScrollAreaImage(QWidget* parent = nullptr);
	void setImage(cv::Mat image);

private:
	cv::Mat m_Image;
	QLabel* m_ImageLabel = nullptr;
	QScrollArea* m_ImageScrollArea;
};


#endif