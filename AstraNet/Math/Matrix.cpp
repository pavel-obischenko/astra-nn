//
//  Matrix.cpp
//  astra-nn
//
//  Created by Pavel on 12/10/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#include "Matrix.hpp"

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
    
    Matrix::Matrix(const StdVectorPtr& data, unsigned long x, unsigned long y, unsigned long width, unsigned long height, unsigned long parentWidth) : x(x), y(y), width(width), height(height), matrixSize(width * height), parentWidth(parentWidth), data(data) {}
    
    Matrix::Matrix(const Matrix& other) : x(0), y(0) {
        width = other.get_width();
        height = other.get_height();
        parentWidth = width;
        
        allocMemory();
        std::copy(other.begin(), other.end(), begin());
    }
    
    Matrix& Matrix::operator=(const Matrix& mat) {
        if (&mat != this) {
            width = mat.get_width();
            height = mat.get_height();
            
            allocMemory();
            std::copy(mat.begin(), mat.end(), begin());
        }
        return *this;
    }
    
    double Matrix::dot_product(const Matrix& mat) const {
        return std::inner_product(begin(), end(), mat.begin(), 0);
    }
    
    Matrix Matrix::element_wise_mul(double arg) const {
        Matrix result(get_width(), get_height());
        std::transform(begin(), end(), result.begin(), std::bind2nd(std::multiplies<double>(), arg));
        return result;
    }

    MatrixPtr Matrix::submatrix(unsigned long x, unsigned long y, unsigned long width, unsigned long height) {
        return MatrixPtr(new Matrix(data, this->x + x, this->y + y, width, height, parentWidth));
    }

    const ConstMatrixPtr Matrix::submatrix(unsigned long x, unsigned long y, unsigned long width, unsigned long height) const {
        return ConstMatrixPtr(new Matrix(data, this->x + x, this->y + y, width, height, parentWidth));
    }
    
    Matrix operator*(const Matrix& left, const Matrix& right) {
        assert(right.get_height() == left.get_width());
        
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
        Matrix result(left.get_width(), left.get_height());
        std::transform(left.begin(), left.end(), right.begin(), result.begin(), std::plus<double>());
        return result;
    }

    Matrix operator-(const Matrix& left, const Matrix& right) {
        Matrix result(left.get_width(), left.get_height());
        std::transform(left.begin(), left.end(), right.begin(), result.begin(), std::minus<double>());
        return result;
    }

    bool operator==(const Matrix& left, const Matrix& right) {
        return *left.get_data_storage() == *right.get_data_storage();
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
    
}}
