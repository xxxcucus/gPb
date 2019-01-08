#ifndef _MPB_GUI_CENTRAL_WIDGET_
#define _MPB_GUI_CENTRAL_WIDGET_

#include <QWidget>
#include <QLabel>
#include "opencv2/opencv.hpp"

class MPbGuiCentralWidget : public QWidget {
	Q_OBJECT

public:
	MPbGuiCentralWidget(QWidget* parent = nullptr);

public slots:
	void loadImage();

private:
	QLabel* m_ImageLabel;
};


#endif