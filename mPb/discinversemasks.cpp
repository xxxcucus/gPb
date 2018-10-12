#include "discinversemasks.h"
#include <QDebug>

DiscInverseMasks::DiscInverseMasks(int radius): m_Radius(radius) {
    computeHalfDiscInfluencePoints();
}

void DiscInverseMasks::computeHalfDiscInfluencePoints() {
    m_InfluencePoints.clear();

    DiscDirectMasks ddm(m_Radius);

    for(int i = -m_Radius; i <= m_Radius; ++i) {
        //qDebug() << "BlaBla " << i;
        for (int j = -m_Radius; j <= m_Radius; ++j) {  //position of the neighbouring points
            std::vector<int> retVal;
            retVal.push_back(i);
            retVal.push_back(j);
            for (int k = 0; k < 8; ++k) {  //arc id
                std::vector<QPoint> vect = ddm.getHalfDiscPoints(k);
                for (auto v : vect) {
                    if ((v.x() == -i) && (v.y() == -j)) {
                        retVal.push_back(k);
                    }
                 }
            }
            if (retVal.size() > 2)
                m_InfluencePoints.push_back(retVal);
            else
                qDebug() << i << " , " << j << " not found ";
        }
    }
}
