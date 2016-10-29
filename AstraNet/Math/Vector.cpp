//
//  Vector.cpp
//  astra-nn
//
//  Created by Pavel on 21/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#include "Vector.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <cassert>

namespace astra {
namespace math {

    Vector::Vector(const std::initializer_list<double>& init) : Matrix(1, init.size()) {
        std::copy(init.begin(), init.end(), begin());
    };

    Vector::Vector(const std::vector<double>& vec) : Matrix(1, vec.size()) {
        std::copy(vec.begin(), vec.end(), begin());
    }

    Vector Vector::element_wise_mul(double arg) const {
        Vector result(size());
        std::transform(begin(), end(), result.begin(), std::bind2nd(std::multiplies<double>(), arg));
        return result;
    }

    Vector Vector::element_wise_mul(const Vector& arg) const {
        assert(size() == arg.size());

        Vector result(size());
        std::transform(begin(), end(), arg.begin(), result.begin(), std::multiplies<double>());
        return result;
    }

    Matrix Vector::toMatrix(unsigned long width, unsigned long height) const {
        assert(width * height == size());

        Matrix result(width, height);
        std::copy(begin(), end(), result.begin());
        return result;
    }

    Vector operator*(const Matrix& left, const Vector& right) {
        assert(right.size() == left.get_width());

        Vector result(left.get_height());
        auto resItr = result.begin();

        left.for_each_row([&right, &resItr](const astra::math::ConstMatrixPtr& row) {
            *resItr++ = row->dot_product(right);
        });
        return result;
    }

    Matrix operator*(const Vector& left, const Matrix& right) {
        return left.transpose() * right;
    }

    Matrix operator*(const Vector& left, const Vector& right) {
        Matrix result(left.size(), right.size());
        auto resItr = result.begin();

        left.for_each([&right, &resItr](double l) {
            right.for_each([l, &resItr](double r) {
                *resItr++ = r * l;
            });
        });
        return result;
    }

    Vector operator*(const Vector& left, double right) {
        return left.element_wise_mul(right);
    }

    Vector operator*(double left, const Vector& right) {
        return right.element_wise_mul(left);
    }

    Vector& operator*=(Vector& left, double right) {
        std::transform(left.begin(), left.end(), left.begin(), std::bind2nd(std::multiplies<double>(), right));
        return left;
    }

    Vector operator+(const Vector& left, const Vector& right) {
        Vector result(left.size());
        std::transform(left.begin(), left.end(), right.begin(), result.begin(), std::plus<double>());
        return result;
    }

    Vector& operator+=(Vector& left, const Vector& right) {
        left = left + right;
        return left;
    }

    Vector operator-(const Vector& left, const Vector& right) {
        Vector result(left.size());
        std::transform(left.begin(), left.end(), right.begin(), result.begin(), std::minus<double>());
        return result;
    }

    Vector& operator-=(Vector& left, const Vector& right) {
        left = left - right;
        return left;
    }

    VectorPtr Vector::head(unsigned long size) {
        return subvec(0, size);
    }

    ConstVectorPtr Vector::head(unsigned long size) const {
        return subvec(0, size);
    }

    VectorPtr Vector::tail(unsigned long size) {
        return subvec(this->size() - size, size - 1);
    }

    ConstVectorPtr Vector::tail(unsigned long size) const {
        return subvec(this->size() - size, size - 1);
    }

    VectorPtr Vector::subvec(unsigned long beginIndex, unsigned long endIndex) {
        return VectorPtr(new Vector(get_data_storage(), beginIndex, endIndex));
    }

    ConstVectorPtr Vector::subvec(unsigned long beginIndex, unsigned long endIndex) const {
        return ConstVectorPtr(new Vector(get_data_storage(), beginIndex, endIndex));
    }
}}
