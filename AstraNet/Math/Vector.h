//
//  Vector.hpp
//  astra-nn
//
//  Created by Pavel on 21/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef Vector_hpp
#define Vector_hpp

#include "Matrix.h"

#include <vector>
#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>

namespace astra {
namespace math {

    class Vector;
    typedef std::shared_ptr<Vector> VectorPtr;
    typedef std::shared_ptr<const Vector> ConstVectorPtr;

    class Vector : public Matrix {
    public:
        Vector() : Matrix() {}
        explicit Vector(unsigned long size) : Matrix(1, size) {}
        Vector(const std::initializer_list<double>& init) : Matrix(1, init.size()) {
            std::copy(init.begin(), init.end(), begin());
        };
        Vector(const Vector& other) : Matrix(other) {}
        explicit Vector(const std::vector<double>& vec) : Matrix(1, vec.size()) {
            std::copy(vec.begin(), vec.end(), begin());
        }
        Vector(const StdVectorPtr& data, unsigned long beginIndex, unsigned long endIndex) : Matrix(data, beginIndex, 0, endIndex - beginIndex, 1, data->size()) {}

        unsigned long size() const { return get_height(); }

        Vector element_wise_mul(double arg) const {
            Vector result(size());
            std::transform(begin(), end(), result.begin(), std::bind2nd(std::multiplies<double>(), arg));
            return result;
        }

        Vector element_wise_mul(const Vector& arg) const {
            assert(size() == arg.size());

            Vector result(size());
            std::transform(begin(), end(), arg.begin(), result.begin(), std::multiplies<double>());
            return result;
        }

        friend Vector operator*(const Matrix& left, const Vector& right) {
            assert(right.size() == left.get_width());

            Vector result(left.get_height());
            auto resItr = result.begin();

            left.for_each_row([&right, &resItr](const astra::math::ConstMatrixPtr& row) {
                *resItr++ = row->dot_product(right);
            });
            return result;
        }

        friend Vector operator*(const Vector& left, const Matrix& right) {
            return right * left;
        }

        friend Matrix operator*(const Vector& left, const Vector& right) {
            Matrix result(left.size(), right.size());
            auto resItr = result.begin();

            left.for_each([&right, &resItr](double l) {
                right.for_each([l, &resItr](double r) {
                    *resItr++ = r * l;
                });
            });
            return result;
        }

        friend Vector operator*(const Vector& left, double right) {
            return left.element_wise_mul(right);
        }
        friend Vector operator*(double left, const Vector& right) {
            return right.element_wise_mul(left);
        }

        friend Vector& operator*=(Vector& left, double right) {
            std::transform(left.begin(), left.end(), left.begin(), std::bind2nd(std::multiplies<double>(), right));
            return left;
        }
        friend Vector operator+(const Vector& left, const Vector& right) {
            Vector result(left.size());
            std::transform(left.begin(), left.end(), right.begin(), result.begin(), std::plus<double>());
            return result;
        }
        friend Vector& operator+=(Vector& left, const Vector& right) {
            left = left + right;
            return left;
        }
        friend Vector operator-(const Vector& left, const Vector& right) {
            Vector result(left.size());
            std::transform(left.begin(), left.end(), right.begin(), result.begin(), std::minus<double>());
            return result;
        }
        friend Vector& operator-=(Vector& left, const Vector& right) {
            left = left - right;
            return left;
        }

        VectorPtr head(unsigned long size) {
            return subvec(0, size - 1);
        }
        ConstVectorPtr head(unsigned long size) const {
            return subvec(0, size - 1);
        }

        VectorPtr tail(unsigned long size) {
            return subvec(this->size() - size, size - 1);
        }
        ConstVectorPtr tail(unsigned long size) const {
            return subvec(this->size() - size, size - 1);
        }

        VectorPtr subvec(unsigned long beginIndex, unsigned long endIndex) {
            return VectorPtr(new Vector(get_data_storage(), beginIndex, endIndex));
        }
        ConstVectorPtr subvec(unsigned long beginIndex, unsigned long endIndex) const {
            return ConstVectorPtr(new Vector(get_data_storage(), beginIndex, endIndex));
        }
    };
}}

#endif /* Vector_hpp */
