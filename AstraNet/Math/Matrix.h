//
//  Matrix.hpp
//  astra-nn
//
//  Created by Pavel on 21/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef Matrix_h
#define Matrix_h

#include "../Common/Iterators.h"

#include <vector>
#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <numeric>
#include <cassert>

namespace astra {
namespace math {
    
    typedef std::shared_ptr<std::vector<double>> StdVectorPtr;
    typedef std::shared_ptr<const std::vector<const double>> StdConstVectorPtr;
    
    template <class _T, class _Function> _Function for_each(const _T* const __t, _Function __f);
    template <class _T, class _Function> _Function for_each(_T* __t, _Function __f);
    
    class Matrix;
    typedef std::shared_ptr<astra::math::Matrix> MatrixPtr;
    typedef std::shared_ptr<const astra::math::Matrix> ConstMatrixPtr;
    
    class Matrix {
    public:
        Matrix() : x(0), y(0), height(0), width(0), matrixSize(0), data(0), parentWidth(0) {}
        Matrix(unsigned long width, unsigned long height);
        Matrix(const std::initializer_list<std::initializer_list<double>>& init);
        Matrix(const Matrix& other);
        Matrix(const StdVectorPtr& data, unsigned long x, unsigned long y, unsigned long width, unsigned long height, unsigned long parentWidth);
    
    public:
        double sum() const { return std::accumulate(begin(), end(), 0.0); }
        Matrix& operator=(const Matrix& mat);
        
        MatrixPtr submatrix(unsigned long x, unsigned long y, unsigned long width, unsigned long height);
        const ConstMatrixPtr submatrix(unsigned long x, unsigned long y, unsigned long width, unsigned long height) const;
        
        Matrix element_wise_mul(double arg) const;
        double dot_product(const Matrix& mat) const;

        Matrix transpose() const;
        
        friend Matrix operator*(const Matrix& left, const Matrix& right);
        friend Matrix operator*(const Matrix& left, double right);
        friend Matrix operator*(double left, const Matrix& right);
        friend Matrix& operator*=(Matrix& left, double right);
        friend Matrix operator+(const Matrix& left, const Matrix& right);
        friend Matrix& operator+=(Matrix& left, const Matrix& right);
        friend Matrix operator-(const Matrix& left, const Matrix& right);
        friend Matrix& operator-=(Matrix& left, const Matrix& right);

        friend bool operator==(const Matrix& left, const Matrix& right);
        
        friend std::ostream& operator<<(std::ostream& os, const Matrix& mat);

        double& operator[](unsigned long index);
        const double& operator[](unsigned long index) const;
        
        template <class _Function> void for_each(_Function __f);
        template <class _Function> void for_each(_Function __f) const;
        
        template <class _Function> void for_each_row(_Function __f);
        template <class _Function> void for_each_row(_Function __f) const;
        
        template <class _Function> void for_each_col(_Function __f);
        template <class _Function> void for_each_col(_Function __f) const;

    public:
        unsigned long size() const { return matrixSize; }
        
        unsigned long get_height() const { return height; }
        unsigned long get_width() const { return width; }

        void debugNaNs() const;

        StdVectorPtr& get_data_storage() { return data; }
        const StdVectorPtr& get_data_storage() const { return data; }

        
    public:
        common::matrix_iterator begin() {
            auto origin = data->begin() + (y * parentWidth + x);
            return common::matrix_iterator(origin, width, height, parentWidth - width);
        }
        common::matrix_iterator end() {
            unsigned long stride = parentWidth - width;
            auto origin = data->begin() + (y * parentWidth + x) + width * height + stride * (height - 1);
            return common::matrix_iterator(origin, width, height, stride);
        }
        
        common::const_matrix_iterator begin() const {
            auto origin = data->begin() + (y * parentWidth + x);
            return common::const_matrix_iterator(origin, width, height, parentWidth - width);
        }
        common::const_matrix_iterator end() const {
            unsigned long stride = parentWidth - width;
            auto origin = data->begin() + (y * parentWidth + x) + width * height + stride * (height - 1);
            return common::const_matrix_iterator(origin, width, height, stride);
        }
        
    protected:
        void set_height(unsigned long r) { height = r; }
        void set_width(unsigned long c) { width = c; }
        
    private:
        void allocMemory() {
            matrixSize = width * height;
            data = std::make_shared<std::vector<double>>(matrixSize);
        }
        
    private:
        unsigned long x, y;
        unsigned long width, height;
        unsigned long parentWidth;
        unsigned long matrixSize;

        
        std::shared_ptr<std::vector<double>> data;
    };
    
    template <class _T, class _Function> _Function for_each(const _T* const __t, _Function __f) {
        return std::for_each(__t->begin(), __t->end(), __f);
    }
    
    template <class _T, class _Function> _Function for_each(_T* __t, _Function __f) {
        return std::for_each(__t->begin(), __t->end(), __f);
    }
    
    template <class _Function> void Matrix::for_each(_Function __f) {
        astra::math::for_each(this, __f);
    }
    
    template <class _Function> void Matrix::for_each(_Function __f) const {
        astra::math::for_each(this, __f);
    }
    
    template <class _Function> void Matrix::for_each_row(_Function __f) {
        for (unsigned int r = 0; r < get_height(); ++r) {
            MatrixPtr m = submatrix(0, r, get_width(), 1);
            __f(m);
        }
    }
    
    template <class _Function> void Matrix::for_each_row(_Function __f) const {
        for (unsigned int r = 0; r < get_height(); ++r) {
            ConstMatrixPtr m = submatrix(0, r, get_width(), 1);
            __f(m);
        }
    }
    
    template <class _Function> void Matrix::for_each_col(_Function __f) {
        for (unsigned int c = 0; c < get_width(); ++c) {
            MatrixPtr m = submatrix(c, 0, 1, get_height());
            __f(m);
        }
    }
    
    template <class _Function> void Matrix::for_each_col(_Function __f) const {
        for (unsigned int c = 0; c < get_width(); ++c) {
            ConstMatrixPtr m = submatrix(c, 0, 1, get_height());
            __f(m);
        }
    }
}}

#endif /* Matrix_h */
