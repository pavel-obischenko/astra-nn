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

#include <vector>
#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>

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
    template <typename T> class MatrixIterator : public std::iterator<std::random_access_iterator_tag, T, ptrdiff_t, T*, T&> {
    public:
        inline MatrixIterator(T* ptr = nullptr) : dataPtr(ptr) {}
        inline MatrixIterator(const MatrixIterator<T>& rawIterator) = default;
        ~MatrixIterator() {}
        
        inline MatrixIterator<T>& operator=(const MatrixIterator<T>& rawIterator) = default;
        inline MatrixIterator<T>& operator=(T* ptr) {
            dataPtr = ptr;
            return *this;
        }
        
        inline operator bool() const {
            return dataPtr ? true : false;
        }
        
        inline bool operator==(const MatrixIterator<T>& rawIterator) const {
            return dataPtr == rawIterator.getConstPtr();
        }
        inline bool operator!=(const MatrixIterator<T>& rawIterator) const {
            return dataPtr != rawIterator.getConstPtr();
        }
        
        inline MatrixIterator<T>& operator+=(const ptrdiff_t& movement) {
            dataPtr += movement;
            return *this;
        }
        inline MatrixIterator<T>& operator-=(const ptrdiff_t& movement) {
            dataPtr -= movement;
            return *this;
        }
        
        inline MatrixIterator<T>& operator++() {
            ++dataPtr;
            return *this;
        }
        inline MatrixIterator<T>& operator--() {
            --dataPtr;
            return *this;
        }
        
        inline MatrixIterator<T> operator++(int) {
            auto temp(*this);
            ++dataPtr;
            return temp;
        }
        inline MatrixIterator<T> operator--(int) {
            auto temp(*this);
            --dataPtr;
            return temp;
        }
        
        inline MatrixIterator<T> operator+(const ptrdiff_t& movement) {
            auto oldPtr = dataPtr;
            dataPtr += movement;
            auto temp(*this);
            dataPtr = oldPtr;
            return temp;
        }
        
        inline MatrixIterator<T> operator-(const ptrdiff_t& movement) {
            auto oldPtr = dataPtr;
            dataPtr -= movement;
            auto temp(*this);
            dataPtr = oldPtr;
            return temp;
        }
        
        inline ptrdiff_t operator-(const MatrixIterator<T>& rawIterator) {
            return std::distance(rawIterator.getPtr(), this->getPtr());
        }
        
        inline T& operator*() {
            return *dataPtr;
        }
        inline const T& operator*() const {
            return *dataPtr;
        }
        inline T* operator->() {
            return dataPtr;
        }
        
        inline T* getPtr() const {
            return dataPtr;
        }
        inline const T* getConstPtr() const {
            return dataPtr;
        }
        
    protected:
        T* dataPtr;
    };
    
    template <typename T> class MatrixReverseIterator : public MatrixIterator<T> {
    public:
        
        inline MatrixReverseIterator(T* ptr = nullptr) : MatrixIterator<T>(ptr) {}
        inline MatrixReverseIterator(const MatrixIterator<T>& rawIterator) {
            this->m_ptr = rawIterator.getPtr();
        }
        inline MatrixReverseIterator(const MatrixReverseIterator<T>& rawReverseIterator) = default;
        ~MatrixReverseIterator() {}
        
        MatrixReverseIterator<T>& operator=(const MatrixReverseIterator<T>& rawReverseIterator) = default;
        MatrixReverseIterator<T>& operator=(const MatrixIterator<T>& rawIterator) {
            this->m_ptr = rawIterator.getPtr();
            return *this;
        }
        MatrixReverseIterator<T>& operator=(T* ptr) {
            this->setPtr(ptr);
            return *this;
        }
        
        MatrixReverseIterator<T>& operator+=(const ptrdiff_t& movement) {
            this->m_ptr -= movement;
            return *this;
        }
        MatrixReverseIterator<T>& operator-=(const ptrdiff_t& movement) {
            this->m_ptr += movement;
            return *this;
        }
        MatrixReverseIterator<T>& operator++() {
            --this->m_ptr;
            return *this;
        }
        MatrixReverseIterator<T>& operator--() {
            ++this->m_ptr;
            return *this;
        }
        MatrixReverseIterator<T> operator++(int) {
            auto temp(*this);
            --this->m_ptr;
            return temp;
        }
        MatrixReverseIterator<T> operator--(int) {
            auto temp(*this);
            ++this->m_ptr;
            return temp;
        }
        MatrixReverseIterator<T> operator+(const int& movement) {
            auto oldPtr = this->m_ptr;
            this->m_ptr-=movement;
            auto temp(*this);
            this->m_ptr = oldPtr;
            return temp;
        }
        MatrixReverseIterator<T> operator-(const int& movement){auto oldPtr = this->m_ptr;this->m_ptr+=movement;auto temp(*this);this->m_ptr = oldPtr;return temp;}
        
        ptrdiff_t operator-(const MatrixReverseIterator<T>& rawReverseIterator) {
            return std::distance(this->getPtr(), rawReverseIterator.getPtr());
        }
        
        MatrixIterator<T> base() {
            MatrixIterator<T> forwardIterator(this->m_ptr);
            ++forwardIterator;
            return forwardIterator;
        }
    };
    
    typedef MatrixIterator<double> iterator;
    typedef MatrixIterator<const double> const_iterator;
    
    typedef MatrixReverseIterator<double>       reverse_iterator;
    typedef MatrixReverseIterator<const double> const_reverse_iterator;
    
    class Matrix {
    public:
        inline Matrix(unsigned long nRows, unsigned long nCols) : nRows(nRows), nCols(nCols), matrixSize(nRows * nCols), data(0) {
            allocMemory();
        }
        
        inline Matrix(const std::initializer_list<std::initializer_list<double>>& init) : Matrix(init.size(), init.begin()->size()) {
            unsigned long index = 0;
            std::for_each(init.begin(), init.end(), [this, &index](const std::initializer_list<double>& item) {
                std::for_each(item.begin(), item.end(), [this, &index](double val) {
                    this->data[index++] = val;
                });
            });
        }
        
        inline Matrix(const Matrix& other) : nRows(other.nRows), nCols(other.nCols), matrixSize(other.matrixSize) {
            allocMemory();
            memcpy(data, other.data, matrixSize * sizeof(double));
        }
        
        inline unsigned long size() const {
            return matrixSize;
        }
        
        inline virtual ~Matrix() {
            if (data) {
                delete [] data;
            }
        }
        
        inline  unsigned long getNRows() const { return nRows; }
        inline unsigned long getNCols() const { return nCols; }
        
        inline iterator begin() {
            return iterator(data);
        }
        inline iterator end() {
            return iterator(&data[size()]);
        }
        
        inline const_iterator cbegin() const {
            return const_iterator(data);
        }
        inline const_iterator cend() const {
            return const_iterator(&data[size()]);
        }
        
        inline reverse_iterator rbegin() {
            return reverse_iterator(&data[size() - 1]);
        }
        inline reverse_iterator rend() {
            return reverse_iterator(&data[-1]);
        }
        
        inline const_reverse_iterator crbegin() const {
            return const_reverse_iterator(&data[size() - 1]);
        }
        inline const_reverse_iterator crend() const {
            return const_reverse_iterator(&data[-1]);
        }
        
    private:
        inline void allocMemory() {
            data = new double[size()];
        }
        
    private:
        unsigned long nCols, nRows;
        unsigned long matrixSize;
        
        double* data;
    };
    
    typedef std::shared_ptr<Matrix> MatrixPtr;
    
}}

#endif /* Matrix_h */
