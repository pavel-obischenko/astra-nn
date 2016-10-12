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
#include "../Common/Iterators.h"

#include <vector>
#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <numeric>

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
    
    template <class _T, class _Function> inline _Function for_each(const _T* const __t, _Function __f);
    template <class _T, class _Function> inline _Function for_each(_T* __t, _Function __f);
    
    class Matrix;
    typedef std::shared_ptr<astra::math::Matrix> MatrixPtr;
    typedef std::shared_ptr<const astra::math::Matrix> ConstMatrixPtr;
    
    class MatrixProxy;
    typedef std::shared_ptr<astra::math::MatrixProxy> MatrixProxyPtr;
    typedef std::shared_ptr<const astra::math::MatrixProxy> ConstMatrixProxyPtr;
    
    class Matrix {
    protected:
        inline Matrix() : height(0), width(0), matrixSize(0), data(0) {}
        
    public:
        Matrix(unsigned long width, unsigned long height);
        Matrix(const std::initializer_list<std::initializer_list<double>>& init);
        Matrix(const Matrix& other);
        Matrix(const MatrixProxy& other);
    
    public:
        inline double sum() const {
            return std::accumulate(begin(), end(), 0);
        }
        
        MatrixProxy submatrix(unsigned long x, unsigned long y, unsigned long width, unsigned long height);
        const MatrixProxy submatrix(unsigned long x, unsigned long y, unsigned long width, unsigned long height) const;
        
        inline Matrix element_wise_mul(double arg) const {
            Matrix result(get_width(), get_height());
            std::transform(begin(), end(), result.begin(), std::bind2nd(std::multiplies<double>(), arg));
            return result;
        }
        
        Matrix& operator=(const Matrix& mat);
        Matrix& operator=(const MatrixProxy& mat);
        
        friend inline Matrix operator*(const Matrix& left, const Matrix& right) {
            Matrix result(right.get_width(), left.get_height());
            
            // TODO:
            
            return result;
        }
        
//        friend inline Vector operator*(const Matrix& left, const Vector& right) {
//            auto mt = left.mul_termwise(right);
//            
//            std::vector<double> result;
//            std::for_each(mt.rows.begin(), mt.rows.end(), [&result](const Vector& row) {
//                result.push_back(row.sum());
//            });
//            return Vector(result);
//        }
//        
//        friend inline Vector operator*(const Vector& left, const Matrix& right) {
//            return right * left;
//        }
        
        friend inline Matrix operator*(const Matrix& left, double right) {
            return left.element_wise_mul(right);
        }
        
        friend inline Matrix operator*(double left, const Matrix& right) {
            return right * left;
        }
        
        friend inline Matrix& operator*=(Matrix& left, double right) {
            std::transform(left.begin(), left.end(), left.begin(), std::bind2nd(std::multiplies<double>(), right));
            return left;
        }
        
        friend inline Matrix operator+(const Matrix& left, const Matrix& right) {
            Matrix result(left.get_width(), left.get_height());
            std::transform(left.begin(), left.end(), right.begin(), result.begin(), std::plus<double>());
            return result;
        }
        
        friend inline Matrix operator-(const Matrix& left, const Matrix& right) {
            Matrix result(left.get_width(), left.get_height());
            std::transform(left.begin(), left.end(), right.begin(), result.begin(), std::minus<double>());
            return result;
        }
        
        friend std::ostream& operator<<(std::ostream& os, const Matrix& mat);
        
        template <class _Function> inline _Function for_each(_Function __f) {
            return astra::math::for_each(this, __f);
        }
        
        template <class _Function> inline _Function for_each(_Function __f) const {
            return astra::math::for_each(this, __f);
        }
        
        template <class _Function> inline _Function for_each_row(_Function __f) {
            auto beginItr = common::stride_iterator(begin(), get_width());
            auto endItr = common::stride_iterator(begin(), 0);
            
            for (auto itr = beginItr; itr != endItr; ++itr) {
                //auto row
            }
            
            return _VSTD::move(__f);
        }

    public:
        inline unsigned long size() const { return matrixSize; }
        
        inline unsigned long get_height() const { return height; }
        inline unsigned long get_width() const { return width; }
        
    public:
        inline common::iterator begin() {
            return common::iterator(data.get());
        }
        inline common::iterator end() {
            return common::iterator(&data.get()[size()]);
        }
        
        inline common::const_iterator begin() const {
            return common::const_iterator(data.get());
        }
        inline common::const_iterator end() const {
            return common::const_iterator(&data.get()[size()]);
        }
        
    protected:
        inline void set_height(unsigned long r) { height = r; }
        inline void set_width(unsigned long c) { width = c; }
        
        inline std::shared_ptr<double>& get_data_ptr() { return data; }
        inline const std::shared_ptr<double>& get_data_ptr() const { return data; }
        
    private:
        inline void allocMemory() {
            matrixSize = width * height;
            data = std::shared_ptr<double>(new double[size()], [](double *p) { delete [] p; });
        }
        
    private:
        unsigned long width, height;
        unsigned long matrixSize;
        
        std::shared_ptr<double> data;
    };
    
    template <class _T, class _Function> inline _Function for_each(const _T* const __t, _Function __f) {
        return std::for_each(__t->begin(), __t->end(), __f);
    }
    
    template <class _T, class _Function> inline _Function for_each(_T* __t, _Function __f) {
        return std::for_each(__t->begin(), __t->end(), __f);
    }
    
    typedef std::shared_ptr<astra::math::Matrix> MatrixPtr;
    typedef std::shared_ptr<const astra::math::Matrix> ConstMatrixPtr;
    
    class Vector : public Matrix {
    protected:
        Vector() : Matrix() {}
        
    public:
        explicit Vector(unsigned long size) : Matrix(1, size) {}
        Vector(const std::initializer_list<std::initializer_list<double>>& init) : Matrix(init) {};
        Vector(const std::initializer_list<double>& init) : Matrix(1, init.size()) {
            unsigned long index = 0;
            double *ptr = get_data_ptr().get();
            std::for_each(init.begin(), init.end(), [&ptr, &index](double val) {
                ptr[index++] = val;
            });
        };
        Vector(const Vector& other) = default;
    };
}}

#endif /* Matrix_h */
