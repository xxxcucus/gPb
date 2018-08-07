#include "texton.h"
#include <cstdlib>
#include <cmath>
#include <stdio.h>

Texton::Texton(int dim): m_Size(dim)
{
    m_Values = std::vector<double>(dim, 0.0);
}

Texton Texton::operator+(const Texton& other) {
    if (m_Size != other.m_Size)
        exit(1);
    Texton retVal(m_Size);
    for (int i = 0; i < m_Size; ++i) {
        retVal.m_Values[i] = other.m_Values[i] + m_Values[i];
    }
    return retVal;
}

Texton Texton::operator/(double val) {
    if (std::abs(val) < 0.00001)
        exit(1);
    Texton retVal(m_Size);
    for (int i = 0; i < m_Size; ++i) {
        retVal.m_Values[i] = m_Values[i] / val;
    }
    return retVal;
}


void Texton::setValueAtIdx(double val, int index) {
    if (index >= m_Size || index < 0)
        return;

    m_Values[index] = val;
}

double Texton::dist(const Texton& t) const {
    if (m_Size != t.m_Size)
        exit(1);
    double retVal = 0;
    for (int i = 0; i < m_Size; ++i) {
        retVal += (m_Values[i] - t.m_Values[i]) * (m_Values[i] - t.m_Values[i]);
    }
    retVal = sqrt(retVal);
    return retVal;
}

void Texton::print() const {
    for (int i = 0; i < m_Size; ++i) {
        printf("%f ", m_Values[i]);
    }
    printf("\n");
}

QString Texton::toString() const {
    QString retVal;
    for (int i = 0; i < m_Size; ++i) {
        retVal += QString::number(m_Values[i]) + " ";
    }
    return retVal;
}
