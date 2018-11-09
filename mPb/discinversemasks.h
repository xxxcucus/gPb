#ifndef DISCINVERSEMASKS_H
#define DISCINVERSEMASKS_H

#include "discdirectmasks.h"

class DiscInverseMasks
{
public:
    DiscInverseMasks(int radius);
    inline std::vector<std::vector<int>>& getHalfDiscInfluencePoints() { return m_InfluencePoints; }

private:
    /**
     * @brief getHalfDiscInfluencePoints - assume histograms are computed in the point (i,j).
     * In which arc sector k is the point (0,0) relative to the point (i,j)
     * @return vector of triples (i, j, k)
     */
    void computeHalfDiscInfluencePoints();


private:
    int m_Radius = 5;
    //vector of vectors with k int elements, where the first two elements are the indices
    //of the neighboring position considered, and the next are the indices of arc sector
    //with center at the neighboring position where the (0,0) is.
    std::vector<std::vector<int>> m_InfluencePoints;

};

#endif // DISCINVERSEMASKS_H
