//
//  Matrix.hpp
//  astra-nn
//
//  Created by Pavel on 21/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef Matrix_h
#define Matrix_h

#include "Vector.hpp"
#include "../Common/Iterators.hpp"

#include <vector>
#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <numeric>
#include <cassert>

namespace astra {
    
    class Matrix {
    public:
        static Matrix oneRowMatrix(const Vector& row) {
            std::vector<Vector> rows; rows.push_back(row);
            return Matrix(rows);
        }
        
        static Matrix oneColMatrix(const Vector& col) {
            std::vector<Vector> rows;
            std::for_each(col.get_storage().begin(), col.get_storage().end(), [&rows](double val) {
                std::vector<double> row; row.push_back(val);
                rows.push_back(Vector(row));
            });
            return Matrix(rows);
        }
        
    public:
        Matrix(unsigned long nRows, unsigned long nCols) : nRows(nRows), nCols(nCols) {
            for (unsigned long i = 0; i < nRows; ++i) {
                rows.emplace_back(nCols);
            }
        }
        
        Matrix(const std::initializer_list<std::initializer_list<double>>& init) {
            std::vector<Vector>& r = rows;
            std::for_each(init.begin(), init.end(), [&r](const std::initializer_list<double>& item) {
                r.emplace_back(std::vector<double>(item));
            });
            
            nRows = init.size();
            nCols = rows.front().size();
        }
        
        explicit Matrix(const std::vector<Vector>& rows) : rows(rows) {
            nRows = rows.size();
            nCols = rows.front().size();
        }
    
        double sum() const {
            double result = 0;
            std::for_each(rows.begin(), rows.end(), [&result](const vec& val) {
                result += val.sum();
            });
            return result;
        }
        
        Matrix mul_termwise(const Vector& vector) const {
            std::vector<Vector> result;
            std::for_each(rows.begin(), rows.end(), [&result, &vector](const Vector& row) {
                result.push_back(row.mul_termwise(vector));
            });
            return Matrix(result);
        }
        
        Matrix mul_termwise(double arg) const {
            std::vector<Vector> result;
            std::for_each(rows.begin(), rows.end(), [&result, arg](const Vector& row) {
                result.push_back(arg * row);
            });
            return Matrix(result);
        }
        
        friend Matrix operator*(const Matrix& left, const Matrix& right) {
            Matrix result(right.nRows, left.nCols);
            
            for (unsigned long col = 0; col < left.nCols; ++col) {
                auto left_col = left.get_col(col);
                
                for (unsigned long row = 0; row < right.nRows; ++row) {
                    auto right_row = right.get_row(row);
                    
                    result.rows[row][col] = left_col.mul_termwise(right_row).sum();
                }
            }
            return result;
        }
        
        friend Vector operator*(const Matrix& left, const Vector& right) {
            auto mt = left.mul_termwise(right);
            
            std::vector<double> result;
            std::for_each(mt.rows.begin(), mt.rows.end(), [&result](const Vector& row) {
                result.push_back(row.sum());
            });
            return Vector(result);
        }
        
        friend Vector operator*(const Vector& left, const Matrix& right) {
            return right * left;
        }
        
        friend Matrix operator*(const Matrix& left, double right) {
            return left.mul_termwise(right);
        }
        
        friend Matrix operator*(double left, const Matrix& right) {
            return right * left;
        }
        
        friend Matrix& operator*=(Matrix& left, double right) {
            left = left * right;
            return left;
        }
        
        friend Matrix operator+(const Matrix& left, const Matrix& right) {
            std::vector<Vector> result;
            for (unsigned int i = 0; i < left.nRows; ++i) {
                result.push_back(left.get_row(i) + right.get_row(i));
            }
            return Matrix(result);
        }
        
        friend std::ostream& operator<<(std::ostream& os, const Matrix& mat) {
            os << "{";
            for(auto item = mat.rows.begin(); item != mat.rows.end(); item++) {
                os << *item << (item != mat.rows.end() - 1 ? ", " : "");
            }
            os << "}";
            return os;
        }
        
        Vector& operator[](unsigned long index) {
            return rows[index];
        }
        
        const std::string to_string() const {
            std::stringstream strm;
            strm << this;
            return strm.str();
        }
        
        Matrix transpose() const {
            std::vector<Vector> cols = get_cols();
            return Matrix(cols);
        }
        
        std::vector<Vector>& get_rows() {
            return rows;
        }
        
        const std::vector<Vector>& get_rows() const {
            return rows;
        }
        
        std::vector<Vector> get_cols() const {
            std::vector<Vector> cols;
            for (unsigned int i = 0; i < nCols; ++i) {
                cols.push_back(get_col(i));
            }
            return cols;
        }
        
        Vector get_row(unsigned long index) const {
            return rows[index];
        }
        
        Vector get_col(unsigned long index) const {
            std::vector<double> result;
            std::for_each(rows.begin(), rows.end(), [&result, index](const Vector& row) {
                result.push_back(row.storage[index]);
            });
            return Vector(result);
        }
        
        unsigned long getNRows() const { return nRows; }
        unsigned long getNCols() const { return nCols; }
        
    protected:
        
        std::vector<Vector> rows;
        unsigned long nCols, nRows;
    };
    
    typedef Matrix mat;
    typedef std::shared_ptr<Matrix> MatrixPtr;
}

namespace astra {
namespace math {
    
    typedef std::shared_ptr<std::vector<double>> StdVectorPtr;
    typedef std::shared_ptr<const std::vector<const double>> StdConstVectorPtr;
    
    template <class _T, class _Function> inline _Function for_each(const _T* const __t, _Function __f);
    template <class _T, class _Function> inline _Function for_each(_T* __t, _Function __f);
    
    class Matrix;
    typedef std::shared_ptr<astra::math::Matrix> MatrixPtr;
    typedef std::shared_ptr<const astra::math::Matrix> ConstMatrixPtr;
    
    class Matrix {
    protected:
        Matrix() : x(0), y(0), height(0), width(0), matrixSize(0), data(0), parentWidth(0) {}
        
    public:
        Matrix(unsigned long width, unsigned long height);
        Matrix(const std::initializer_list<std::initializer_list<double>>& init);
        Matrix(const Matrix& other);
        Matrix(const StdVectorPtr& data, unsigned long x, unsigned long y, unsigned long width, unsigned long height, unsigned long parentWidth);
    
    public:
        double sum() const { return std::accumulate(begin(), end(), 0); }
        Matrix& operator=(const Matrix& mat);
        
        MatrixPtr submatrix(unsigned long x, unsigned long y, unsigned long width, unsigned long height);
        const ConstMatrixPtr submatrix(unsigned long x, unsigned long y, unsigned long width, unsigned long height) const;
        
        Matrix element_wise_mul(double arg) const;
        double dot_product(const Matrix& mat) const;
        
        friend Matrix operator*(const Matrix& left, const Matrix& right);
        friend Matrix operator*(const Matrix& left, double right);
        friend Matrix operator*(double left, const Matrix& right);
        friend Matrix& operator*=(Matrix& left, double right);
        friend Matrix operator+(const Matrix& left, const Matrix& right);
        friend Matrix operator-(const Matrix& left, const Matrix& right);

        friend bool operator==(const Matrix& left, const Matrix& right);
        
        friend std::ostream& operator<<(std::ostream& os, const Matrix& mat);
        
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
        
        StdVectorPtr& get_data_storage() { return data; }
        const StdVectorPtr& get_data_storage() const { return data; }
        
    private:
        void allocMemory() {
            matrixSize = width * height;
            data = std::make_shared<std::vector<double>>(matrixSize);
        }
        
    private:
        unsigned long x, y;
        unsigned long width, height;
        unsigned long matrixSize;
        unsigned long parentWidth;
        
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
    
    typedef std::shared_ptr<astra::math::Matrix> MatrixPtr;
    typedef std::shared_ptr<const astra::math::Matrix> ConstMatrixPtr;
    
    // ********************* Vector **********************
    
    class Vector : public Matrix {
    protected:
        Vector() : Matrix() {}
        
    public:
        explicit Vector(unsigned long size) : Matrix(1, size) {}
        Vector(const std::initializer_list<double>& init) : Matrix(1, init.size()) {
            std::copy(init.begin(), init.end(), begin());
        };
        Vector(const Vector& other) : Matrix(other) {}

        unsigned long size() const {
            return get_height();
        }

        friend inline Vector operator*(const Matrix& left, const Vector& right) {
            assert(right.size() == left.get_width());

            Vector result(left.get_height());
            auto resItr = result.begin();

            left.for_each_row([&right, &resItr](const astra::math::ConstMatrixPtr& row) {
                *resItr++ = row->dot_product(right);
            });

            return result;
        }

        friend inline Vector operator*(const Vector& left, const Matrix& right) {
            return right * left;
        }
    };
}}

#endif /* Matrix_h */
