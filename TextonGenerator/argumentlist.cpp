#include "argumentlist.h"

void ArgumentList::argsToStringlist(int argc, char * argv []) {
	for (int i=0; i < argc; ++i) {
		*this += argv[i];
	}
}

bool ArgumentList::getSwitch(const QString& option) {
	QMutableStringListIterator itr(*this);
	while (itr.hasNext()) {
		if (option == itr.next()) {
			itr.remove();
			return true;
		}
	}
	return false;
}

QString ArgumentList::getSwitchArg(const QString& option, const QString& defaultValue) {
	if (isEmpty())
		return defaultValue;
	QMutableStringListIterator itr(*this);
	while (itr.hasNext()) {
		if (option == itr.next()) {
			itr.remove();
			QString retval = itr.next();
			itr.remove();
			return retval;
		}
	}
	return defaultValue;
}