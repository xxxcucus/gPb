#ifndef __MPB_GUI_MAIN_WINDOW__
#define __MPB_GUI_MAIN_WINDOW__

#include <QMainWindow>
#include <QLabel>
#include "mpbguicentralwidget.h"

class MPbGuiMainWindow : public QMainWindow {

	Q_OBJECT

public:
	explicit MPbGuiMainWindow(QWidget* parent = nullptr);

private:
	MPbGuiCentralWidget* m_CentralWidget;
};


#endif