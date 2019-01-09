#ifndef _MPB_GUI_CENTRAL_WIDGET_
#define _MPB_GUI_CENTRAL_WIDGET_

#include <QTabWidget>
#include <QLabel>
#include "opencv2/opencv.hpp"
#include "scrollareaimage.h"

class MPbGuiCentralWidget : public QTabWidget {
	Q_OBJECT

public:
	MPbGuiCentralWidget(QWidget* parent = nullptr);

public slots:
	void loadImage();

private:
	ScrollAreaImage* m_ScrollAreaImage;
	ScrollAreaImage* m_ScrollAreaEdges;
	cv::Mat m_Image;
	cv::Mat m_EdgesImage;
};


#endif