#ifndef TEXTON_H
#define TEXTON_H

#include <QString>
#include <vector>

/**
 * @brief 17-dimensional vector
 */
class Texton
{
public:
    Texton(int dim = 17);

    Texton operator+(const Texton&);
    Texton operator/(double val);
    double dist(const Texton&) const;
    void setValueAtIdx(double val, int index);
    void print() const;
    QString toString() const;

private:
    std::vector<double> m_Values;
    int m_Size = 17;
};

#endif // TEXTON_H
