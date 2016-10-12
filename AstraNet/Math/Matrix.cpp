//
//  Matrix.cpp
//  astra-nn
//
//  Created by Pavel on 12/10/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#include "Matrix.hpp"
#include "MatrixProxy.hpp"

namespace astra {
namespace math {
    
    Matrix::Matrix(unsigned long width, unsigned long height) : width(width), height(height), matrixSize(0), data(0) {
        allocMemory();
    }
    
    Matrix::Matrix(const std::initializer_list<std::initializer_list<double>>& init) : Matrix(init.begin()->size(), init.size()) {
        unsigned long index = 0;
        std::for_each(init.begin(), init.end(), [this, &index](const std::initializer_list<double>& item) {
            std::for_each(item.begin(), item.end(), [this, &index](double val) {
                this->data.get()[index++] = val;
            });
        });
    }
    
    Matrix::Matrix(const Matrix& other) {
        width = other.get_width();
        height = other.get_height();
        
        allocMemory();
        std::copy(other.begin(), other.end(), begin());
    }
    
    Matrix::Matrix(const MatrixProxy& other) {
        width = other.get_width();
        height = other.get_height();
        
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
    
    Matrix& Matrix::operator=(const MatrixProxy& mat) {
        if (&mat != this) { // TODO: &mat.originMat != this
            width = mat.get_width();
            height = mat.get_height();
            
            allocMemory();
            std::copy(mat.begin(), mat.end(), begin());
        }
        return *this;
    }

    MatrixProxy Matrix::submatrix(unsigned long x, unsigned long y, unsigned long width, unsigned long height) {
        common::iterator itr = begin().operator +(y * get_width() + x);
        return MatrixProxy(itr.getPtr(), width, height, get_width());
    }
    
    const MatrixProxy Matrix::submatrix(unsigned long x, unsigned long y, unsigned long width, unsigned long height) const {
        common::const_iterator itr = begin().operator +(y * get_width() + x);
        return MatrixProxy(const_cast<double*>(itr.getPtr()), width, height, get_width());
    }
    
    std::ostream& operator<<(std::ostream& os, const Matrix& mat) {
        unsigned long i = 0;
        unsigned long w = mat.get_width();
        
        os << "\n ";
        mat.for_each([&os, &i, w](double val) {
            os << val << ((i > 0 && (i + 1) % w == 0) ? "\n " : " ");
            ++i;
        });
        os << "\n";
        return os;
    }
    
}}
