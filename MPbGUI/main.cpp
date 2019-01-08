#include "mpbguimainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);
	MPbGuiMainWindow* mgmw = new MPbGuiMainWindow();
	mgmw->show();
	return app.exec();
}
