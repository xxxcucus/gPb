#include "discdirectmasks.h"
#include <utility>
#include <cmath>
#include <QDebug>

DiscDirectMasks::DiscDirectMasks(int radius): m_Radius(radius)
{
    generateArcs();
    generateBorders();
}

void DiscDirectMasks::generateArcs() {

    m_Counters = std::vector<int>(8, 0);

    for (int i = -m_Radius; i <= m_Radius; ++i) {
        for (int j = -m_Radius; j <= m_Radius; ++j) {
            if (i == 0 && j == 0)
                continue;
            if (i * i + j * j > m_Radius * m_Radius)
                continue;
            QPoint pt(i, j);
            if (i < 0 && j >= 0) {
                if (-i > j) {
                    m_MaskPoints.insert(std::make_pair(0, pt));
                    m_Counters[1]++;
                } else {
                    m_MaskPoints.insert(std::make_pair(1, pt));
                    m_Counters[2]++;
                }
            }

            if (i >= 0 && j > 0) {
                if (i < j) {
                    m_MaskPoints.insert(std::make_pair(2, pt));
                    m_Counters[3]++;
                } else {
                    m_MaskPoints.insert(std::make_pair(3, pt));
                    m_Counters[4]++;
                }
            }

           if (j <= 0 && i > 0) {
                if (i > -j) {
                    m_MaskPoints.insert(std::make_pair(4, pt));
                    m_Counters[5]++;
                } else {
                    m_MaskPoints.insert(std::make_pair(5, pt));
                    m_Counters[6]++;
                }
            }

            if (i <= 0 && j < 0) {
                if (-i < -j) {
                    m_MaskPoints.insert(std::make_pair(6, pt));
                   m_Counters[7]++;
                } else  {
                    m_MaskPoints.insert(std::make_pair(7, pt));
                    m_Counters[8]++;
                }
            }
        }
    }
}

void DiscDirectMasks::generateBorders() {
    for (int i = 1; i <= m_Radius; ++i) {
        QPoint pt(i, 0);
        m_BorderPoints.insert(std::make_pair(0, pt));
    }
    for (int i = 1; i <= double(m_Radius)/sqrt(2.0); ++i) {
        QPoint pt(i, -i);
        m_BorderPoints.insert(std::make_pair(1, pt));
    }
    for (int i = 1; i <= m_Radius; ++i) {
        QPoint pt(0, -i);
        m_BorderPoints.insert(std::make_pair(2, pt));
    }
    for (int i = 1; i <= double(m_Radius)/sqrt(2.0); ++i) {
        QPoint pt(-i, -i);
        m_BorderPoints.insert(std::make_pair(3, pt));
    }
    for (int i = 1; i <= m_Radius; ++i) {
        QPoint pt(-i, 0);
        m_BorderPoints.insert(std::make_pair(4, pt));
    }
    for (int i = 1; i <= double(m_Radius)/sqrt(2.0); ++i) {
        QPoint pt(-i, i);
        m_BorderPoints.insert(std::make_pair(5, pt));
    }
    for (int i = 1; i <= m_Radius; ++i) {
        QPoint pt(0, i);
        m_BorderPoints.insert(std::make_pair(6, pt));
    }
    for (int i = 1; i <= double(m_Radius)/sqrt(2.0); ++i) {
        QPoint pt(i, i);
        m_BorderPoints.insert(std::make_pair(7, pt));
    }
}

std::vector<QPoint> DiscDirectMasks::getArcPoints(int arcId) {
    auto range = m_MaskPoints.equal_range(arcId);
    std::vector<QPoint> retVal;

    for (auto it = range.first; it != range.second; ++it) {
        retVal.push_back(QPoint(it->second.x(), it->second.y()));
    }

    return retVal;
}

std::vector<QPoint> DiscDirectMasks::getHalfDiscPoints(int arcId) {
    std::vector<QPoint> retVal;
    for (int i = 0; i < 4; ++i) {
        int tArcId = (arcId + i) % 8;
        auto range = m_MaskPoints.equal_range(tArcId);
        for (auto it = range.first; it != range.second; ++it) {
            retVal.push_back(it->second);
        }
    }

    auto range1 = m_BorderPoints.equal_range(arcId);
    for (auto it = range1.first; it != range1.second; ++it) {
        retVal.push_back(it->second);
    }
    return retVal;
}

