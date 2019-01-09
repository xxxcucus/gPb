#include "mpbguimainwindow.h"
#include <QMenu>
#include <QMenuBar>

MPbGuiMainWindow::MPbGuiMainWindow(QWidget* parent) : QMainWindow(parent) {
	QMenu* fileMenu;
	fileMenu = menuBar()->addMenu("&File");
	
	QAction* loadAction = new QAction("&Load Image", this);
	fileMenu->addAction(loadAction);

	m_CentralWidget = new MPbGuiCentralWidget();
	connect(loadAction, SIGNAL(triggered(bool)), m_CentralWidget, SLOT(loadImage()));
	setCentralWidget(m_CentralWidget);

	//setFixedSize(3000, 2000);
}

