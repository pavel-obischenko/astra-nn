//
//  Vector.h
//  astra-nn
//
//  Created by Pavel on 21/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef Vector_hpp
#define Vector_hpp

#include "Matrix.h"
#include <vector>

namespace astra {
namespace math {

    class Vector;
    typedef std::shared_ptr<Vector> VectorPtr;
    typedef std::shared_ptr<const Vector> ConstVectorPtr;

    class Vector : public Matrix {
    public:
        Vector() : Matrix() {}
        explicit Vector(unsigned long size) : Matrix(1, size) {}
        Vector(const std::initializer_list<double>& init);
        Vector(const Vector& other) : Matrix(other) {}
        explicit Vector(const std::vector<double>& vec);
        Vector(const StdVectorPtr& data, unsigned long beginIndex, unsigned long endIndex) : Matrix(data, beginIndex, 0, 1, endIndex - beginIndex, 1) {}

        unsigned long size() const { return get_height(); }

        Vector element_wise_mul(double arg) const;
        Vector element_wise_mul(const Vector& arg) const;

        friend Vector operator*(const Matrix& left, const Vector& right);
        friend Matrix operator*(const Vector& left, const Matrix& right);
        friend Matrix operator*(const Vector& left, const Vector& right);
        friend Vector operator*(const Vector& left, double right);
        friend Vector operator*(double left, const Vector& right);
        friend Vector& operator*=(Vector& left, double right);
        friend Vector operator+(const Vector& left, const Vector& right);
        friend Vector& operator+=(Vector& left, const Vector& right);
        friend Vector operator-(const Vector& left, const Vector& right);
        friend Vector& operator-=(Vector& left, const Vector& right);

        VectorPtr head(unsigned long size);
        ConstVectorPtr head(unsigned long size) const;

        VectorPtr tail(unsigned long size);
        ConstVectorPtr tail(unsigned long size) const;

        VectorPtr subvec(unsigned long beginIndex, unsigned long endIndex);
        ConstVectorPtr subvec(unsigned long beginIndex, unsigned long endIndex) const;
    };
}}

#endif /* Vector_hpp */
