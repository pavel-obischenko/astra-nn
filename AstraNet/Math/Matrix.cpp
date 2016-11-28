//
//  Matrix.cpp
//  astra-nn
//
//  Created by Pavel on 12/10/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#include "Matrix.h"
#include "Vector.h"

#include <algorithm>
#include <functional>
#include <sstream>
#include <numeric>
#include <cassert>
#include <random>

namespace astra {
namespace math {
    
    Matrix::Matrix(unsigned long width, unsigned long height) : x(0), y(0), width(width), height(height), matrixSize(0), parentWidth(width), data(0) {
        allocMemory();
    }
    
    Matrix::Matrix(const std::initializer_list<std::initializer_list<double>>& init) : Matrix(init.begin()->size(), init.size()) {
        auto itr = begin();

        std::for_each(init.begin(), init.end(), [&itr](const std::initializer_list<double>& item) {
            std::copy(item.begin(), item.end(), itr);
            itr += item.end() - item.begin();
        });
    }

    Matrix::Matrix(const Matrix& other) : x(0), y(0) {
        width = other.get_width();
        height = other.get_height();
        parentWidth = width;

        allocMemory();
        std::copy(other.begin(), other.end(), begin());
    }
    
    Matrix::Matrix(const StdVectorPtr& data, unsigned long x, unsigned long y, unsigned long width, unsigned long height, unsigned long parentWidth) : x(x), y(y), width(width), height(height), matrixSize(width * height), parentWidth(parentWidth), data(data) {}

    Matrix::Matrix(const StdVectorPtr& data, unsigned long width, unsigned long height) : x(0), y(0), width(width), height(height), parentWidth(width) {
        allocMemory();
        std::copy(data->begin(), data->end(), begin());
    }

    Matrix Matrix::zero(unsigned long width, unsigned long height) {
        Matrix result(width, height);
        result.zeroFill();
        return result;
    }

    Matrix Matrix::identity(unsigned long dimensions) {
        Matrix result(dimensions, dimensions);

        unsigned long diagonalIndex = 0;
        unsigned long size = dimensions * dimensions;
        for (unsigned long i = 0; i < size; ++i) {
            if (i != diagonalIndex) {
                result[i] = 0;
            }
            else {
                result[i] = 1;
                diagonalIndex += dimensions + 1;
            }
        }
        return result;
    }

    Matrix Matrix::rnd(unsigned long width, unsigned long height, double min, double max) {
        Matrix result(width, height);
        result.rndFill(min, max);
        return result;
    }

    Matrix Matrix::copy(const Matrix& other, unsigned long padWidth, unsigned long padHeight) {
        Matrix result = Matrix::zero(other.get_width() + 2 * padWidth, other.get_height() + 2 * padHeight);

        MatrixPtr submatrix = result.submatrix(padWidth, padHeight, other.get_width(), other.get_height());
        std::copy(other.begin(), other.end(), submatrix->begin());

        return result;
    }

    void Matrix::fill(double arg) {
        for_each([arg](double &val) {
            val = arg;
        });
    }

    void Matrix::zeroFill() {
        fill(0);
    }

    void Matrix::rndFill(double min, double max) {
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(min, max);
        auto rnd = std::bind(distribution, generator);

        for_each([&rnd](double &val) {
            val = rnd();
        });
    }

    double Matrix::sum() const {
        return std::accumulate(begin(), end(), 0.0);
    }

    double Matrix::max_element() const {
        return *std::max_element(begin(), end());
    }

    double Matrix::average_value() const {
        return sum() / (get_width() * get_height());
    }

    Matrix& Matrix::operator=(const Matrix& mat) {
        if (&mat != this) {
            width = mat.get_width();
            height = mat.get_height();
            parentWidth = mat.parentWidth;

            allocMemory();
            std::copy(mat.begin(), mat.end(), begin());
        }
        return *this;
    }
    
    double Matrix::dot_product(const Matrix& mat) const {
        return std::inner_product(begin(), end(), mat.begin(), 0.0);
    }
    
    Matrix Matrix::element_wise_mul(double arg) const {
        Matrix result(get_width(), get_height());
        std::transform(begin(), end(), result.begin(), std::bind2nd(std::multiplies<double>(), arg));
        return result;
    }

    Matrix Matrix::element_wise_mul(const Matrix& mat) const {
        assert(get_width() == mat.get_width() && get_height() == mat.get_height());

        Matrix result(get_width(), get_height());
        std::transform(begin(), end(), mat.begin(), result.begin(), std::multiplies<double>());
        return result;
    }

    Matrix Matrix::transpose() const {
        Matrix result(get_height(), get_width());

        auto src = data->begin();
        auto dst = result.begin();

        for (int i = 0; i < get_width(); ++i, ++src) {
            common::const_stride_iterator col(src, get_width());

            for (int j = 0; j < get_height(); ++j, ++col, ++dst) {
                *dst = *col;
            }
        }
        return result;
    }

    MatrixPtr Matrix::submatrix(unsigned long x, unsigned long y, unsigned long width, unsigned long height) {
        return MatrixPtr(new Matrix(data, this->x + x, this->y + y, width, height, parentWidth));
    }

    const ConstMatrixPtr Matrix::submatrix(unsigned long x, unsigned long y, unsigned long width, unsigned long height) const {
        return ConstMatrixPtr(new Matrix(data, this->x + x, this->y + y, width, height, parentWidth));
    }

    Vector Matrix::toVector() const {
        return Vector(*this);
    }
    
    Matrix operator*(const Matrix& left, const Matrix& right) {
        assert(left.get_width() == right.get_height());
        
        Matrix result(right.get_width(), left.get_height());
        auto resItr = result.begin();
        
        left.for_each_row([&right, &resItr](const astra::math::ConstMatrixPtr& row) {
            right.for_each_col([&row, &resItr](const astra::math::ConstMatrixPtr& col) {
                *resItr++ = row->dot_product(*col);
            });
        });
        return result;
    }

    Matrix operator*(const Matrix& left, double right) {
        return left.element_wise_mul(right);
    }

    Matrix operator*(double left, const Matrix& right) {
        return right * left;
    }

    Matrix& operator*=(Matrix& left, double right) {
        std::transform(left.begin(), left.end(), left.begin(), std::bind2nd(std::multiplies<double>(), right));
        return left;
    }

    Matrix operator+(const Matrix& left, const Matrix& right) {
        assert(left.get_width() == right.get_width() && left.get_height() == right.get_height());

        Matrix result(left.get_width(), left.get_height());
        std::transform(left.begin(), left.end(), right.begin(), result.begin(), std::plus<double>());
        return result;
    }

    Matrix& operator+=(Matrix& left, const Matrix& right) {
        left = left + right;
        return left;
    }

    Matrix operator-(const Matrix& left, const Matrix& right) {
        assert(left.get_width() == right.get_width() && left.get_height() == right.get_height());

        Matrix result(left.get_width(), left.get_height());
        std::transform(left.begin(), left.end(), right.begin(), result.begin(), std::minus<double>());
        return result;
    }

    Matrix& operator-=(Matrix& left, const Matrix& right) {
        left = left - right;
        return left;
    }

    bool operator==(const Matrix& left, const Matrix& right) {
        return *left.get_data_storage() == *right.get_data_storage();
    }

    double& Matrix::operator[](unsigned long index) {
        return (*get_data_storage())[index];
    }

    const double& Matrix::operator[](unsigned long index) const {
        return (*get_data_storage())[index];
    }
    
    std::ostream& operator<<(std::ostream& os, const Matrix& mat) {
        unsigned long i = 0;
        unsigned long w = mat.get_width();
        os << " ";
        mat.for_each([&os, &i, w](double val) {
            bool newLine = (i > 0 && (i + 1) % w == 0) || w == 1;
            os << val << (newLine ? "\n " : " ");
            ++i;
        });
        return os;
    }

    void Matrix::allocMemory() {
        matrixSize = width * height;
        data = std::make_shared<std::vector<double>>(matrixSize);
    }

    void Matrix::debugNaNs() const {
        for_each([] (double v) {
            assert(!std::isnan(v));
        });
    }
}}
