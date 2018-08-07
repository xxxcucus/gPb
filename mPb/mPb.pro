TEMPLATE = app
CONFIG += c++11
CONFIG += qt

SOURCES += \
    main.cpp \
    pbdetector.cpp \
    discdirectmasks.cpp \
    discinversemasks.cpp

INCLUDEPATH += "D:\temp\opencv\sources\build\install\include"

LIBS += -L"D:\temp\opencv\sources\build\install\x86\mingw\lib" \
    -lopencv_core320.dll \
    -lopencv_highgui320.dll \
    -lopencv_imgcodecs320.dll \
    -lopencv_imgproc320.dll \
    -lopencv_features2d320.dll \
    -lopencv_calib3d320.dll \
    -lopencv_videoio320.dll

HEADERS += \
    pbdetector.h \
    discdirectmasks.h \
    discinversemasks.h


win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../common/release/ -lcommon
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../common/debug/ -lcommon
else:unix: LIBS += -L$$OUT_PWD/../common/ -lcommon

INCLUDEPATH += $$PWD/../common
DEPENDPATH += $$PWD/../common
