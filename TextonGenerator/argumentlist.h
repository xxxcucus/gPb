#ifndef ARGUMENTLIST_H
#define ARGUMENTLIST_H

#include <QStringList>

class ArgumentList : public QStringList {

public:
	ArgumentList(int argc, char* argv[]) {
		argsToStringlist(argc, argv);
	}
	ArgumentList(const QStringList& argumentList): QStringList(argumentList) {}
	
	bool getSwitch(const QString& option);
	QString getSwitchArg(const QString& option, const QString& defaultRetValue=QString());

private:
	void argsToStringlist(int argc, char* argv[]);
};
#endif