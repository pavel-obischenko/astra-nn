//
//  Iterators.h
//  astra-nn
//
//  Created by Pavel on 11/10/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef Iterators_h
#define Iterators_h

#include <iterator>
#include <vector>
#include <cmath>
#include <cstdlib>

namespace astra {
namespace common {
    
    template <typename T> class RawIterator : public std::iterator<std::random_access_iterator_tag, T, ptrdiff_t, T*, T&> {
    public:
        inline RawIterator(T* ptr = nullptr) : dataPtr(ptr) {}
        inline RawIterator(const RawIterator<T>& rawIterator) = default;
        ~RawIterator() {}
        
        inline RawIterator<T>& operator=(const RawIterator<T>& rawIterator) = default;
        inline RawIterator<T>& operator=(T* ptr) {
            dataPtr = ptr;
            return *this;
        }
        
        inline operator bool() const {
            return dataPtr ? true : false;
        }
        
        inline bool operator==(const RawIterator<T>& rawIterator) const {
            return dataPtr == rawIterator.getConstPtr();
        }
        inline bool operator!=(const RawIterator<T>& rawIterator) const {
            return dataPtr != rawIterator.getConstPtr();
        }
        
        inline RawIterator<T>& operator+=(const long movement) {
            dataPtr += movement;
            return *this;
        }
        inline RawIterator<T>& operator-=(const long movement) {
            dataPtr -= movement;
            return *this;
        }
        
        inline RawIterator<T>& operator++() {
            ++dataPtr;
            return *this;
        }
        inline RawIterator<T>& operator--() {
            --dataPtr;
            return *this;
        }
        
        inline RawIterator<T> operator++(int) {
            auto temp(*this);
            ++dataPtr;
            return temp;
        }
        inline RawIterator<T> operator--(int) {
            auto temp(*this);
            --dataPtr;
            return temp;
        }
        
        inline RawIterator<T> operator+(const long movement) {
            auto oldPtr = dataPtr;
            dataPtr += movement;
            auto temp(*this);
            dataPtr = oldPtr;
            return temp;
        }
        
        inline RawIterator<T> operator-(const long movement) {
            auto oldPtr = dataPtr;
            dataPtr -= movement;
            auto temp(*this);
            dataPtr = oldPtr;
            return temp;
        }
        
        inline ptrdiff_t operator-(const RawIterator<T>& rawIterator) {
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
        inline const T* const getConstPtr() const {
            return dataPtr;
        }
        
    protected:
        T* dataPtr;
    };
    
    template <typename T> class RawReverseIterator : public RawIterator<T> {
    public:
        inline RawReverseIterator(T* ptr = nullptr) : RawIterator<T>(ptr) {}
        inline RawReverseIterator(const RawIterator<T>& rawIterator) {
            this->dataPtr = rawIterator.getPtr();
        }
        inline RawReverseIterator(const RawReverseIterator<T>& rawReverseIterator) = default;
        ~RawReverseIterator() {}
        
        RawReverseIterator<T>& operator=(const RawReverseIterator<T>& rawReverseIterator) = default;
        RawReverseIterator<T>& operator=(const RawIterator<T>& rawIterator) {
            this->dataPtr = rawIterator.getPtr();
            return *this;
        }
        RawReverseIterator<T>& operator=(T* ptr) {
            this->setPtr(ptr);
            return *this;
        }
        
        RawReverseIterator<T>& operator+=(const long movement) {
            this->dataPtr -= movement;
            return *this;
        }
        RawReverseIterator<T>& operator-=(const long movement) {
            this->dataPtr += movement;
            return *this;
        }
        RawReverseIterator<T>& operator++() {
            --this->dataPtr;
            return *this;
        }
        RawReverseIterator<T>& operator--() {
            ++this->dataPtr;
            return *this;
        }
        RawReverseIterator<T> operator++(int) {
            auto temp(*this);
            --this->dataPtr;
            return temp;
        }
        RawReverseIterator<T> operator--(int) {
            auto temp(*this);
            ++this->dataPtr;
            return temp;
        }
        RawReverseIterator<T> operator+(const long movement) {
            auto oldPtr = this->dataPtr;
            this->dataPtr -= movement;
            auto temp(*this);
            this->dataPtr = oldPtr;
            return temp;
        }
        RawReverseIterator<T> operator-(const long movement) {
            auto oldPtr = this->dataPtr;
            this->dataPtr += movement;
            auto temp(*this);
            this->dataPtr = oldPtr;
            return temp;
        }
        
        ptrdiff_t operator-(const RawReverseIterator<T>& rawReverseIterator) {
            return std::distance(this->getPtr(), rawReverseIterator.getPtr());
        }
        
        RawIterator<T> base() {
            RawIterator<T> forwardIterator(this->dataPtr);
            ++forwardIterator;
            return forwardIterator;
        }
    };

    template <typename T, class Itr> class StrideIteratorAdapter : public std::iterator<std::random_access_iterator_tag, T, ptrdiff_t, T*, T&> {
    public:
        explicit StrideIteratorAdapter(const Itr& itr, unsigned long stride) : itr(itr), stride(stride) {}
        inline StrideIteratorAdapter(const StrideIteratorAdapter<T, Itr>& adapter) = default;
        ~StrideIteratorAdapter() {}
        
        inline StrideIteratorAdapter<T, Itr>& operator=(const StrideIteratorAdapter<T, Itr>& adapter) = default;
        
        inline operator bool() const {
            return itr.operator bool();
        }
        
        inline bool operator==(const StrideIteratorAdapter<T, Itr>& other) const {
            return itr.operator ==(other.itr);
        }
        inline bool operator!=(const StrideIteratorAdapter<T, Itr>& other) const {
            return itr.operator !=(other.itr);
        }
        
        inline StrideIteratorAdapter<T, Itr>& operator+=(const long movement) {
            itr.operator +=(movement * stride);
            return *this;
        }
        inline StrideIteratorAdapter<T, Itr>& operator-=(const long movement) {
            itr.operator -=(movement * stride);
            return *this;
        }
        
        inline StrideIteratorAdapter<T, Itr>& operator++() {
            return operator +=(1);
        }
        inline StrideIteratorAdapter<T, Itr>& operator--() {
            return operator -=(1);
        }
        
        inline StrideIteratorAdapter<T, Itr> operator++(int) {
            auto temp(*this);
            operator ++();
            return temp;
        }
        inline StrideIteratorAdapter<T, Itr> operator--(int) {
            auto temp(*this);
            operator --();
            return temp;
        }
        
        inline StrideIteratorAdapter<T, Itr> operator+(const long movement) {
            auto temp(*this);
            temp += movement;
            return temp;
        }
        
        inline StrideIteratorAdapter<T, Itr> operator-(const long movement) {
            auto temp(*this);
            temp -= movement;
            return temp;
        }
        
        inline ptrdiff_t operator-(const StrideIteratorAdapter<T, Itr>& strideIterator) {
            return std::distance(strideIterator.getPtr(), this->getPtr());
        }
        
        inline T& operator*() {
            return itr.operator *();
        }
        inline const T& operator*() const {
            return itr.operator *();
        }
        inline T* operator->() {
            return itr.operator ->();
        }
        
        inline T* getPtr() const {
            return itr.getPtr();
        }
        inline const T* getConstPtr() const {
            return itr.getConstPtr();
        }
        
    private:
        inline StrideIteratorAdapter<T, Itr>& operator=(T* ptr) { return *this; }
        
    protected:
        Itr itr;
        unsigned long stride;
    };
    
    typedef RawIterator<double> iterator;
    typedef RawIterator<const double> const_iterator;
    
    typedef StrideIteratorAdapter<double, std::vector<double>::iterator> stride_iterator;
    typedef StrideIteratorAdapter<const double, std::vector<const double>::const_iterator> const_stride_iterator;
    
    typedef RawReverseIterator<double>       reverse_iterator;
    typedef RawReverseIterator<const double> const_reverse_iterator;
    
    // ******************** MatrixIteratorAdapter ********************
    
    template <typename T, class Itr> class MatrixIteratorAdapter : public std::iterator<std::random_access_iterator_tag, T, ptrdiff_t, T*, T&> {
    public:
        inline MatrixIteratorAdapter(const Itr& origin, unsigned long width, unsigned long height, unsigned long stride) : ip(0), max_ip(width * height), itr(origin), width(width), height(height), stride(stride) {}
        
        inline MatrixIteratorAdapter(const MatrixIteratorAdapter<T, Itr>& adapter) = default;
        ~MatrixIteratorAdapter() {}
        
        inline MatrixIteratorAdapter<T, Itr>& operator=(const MatrixIteratorAdapter<T, Itr>& adapter) = default;
        
        inline operator bool() const {
            return itr.operator bool();
        }
        
        inline bool operator ==(const MatrixIteratorAdapter<T, Itr>& other) const {
            return itr == other.itr;
        }
        inline bool operator !=(const MatrixIteratorAdapter<T, Itr>& other) const {
            return itr != other.itr;
        }
        
        inline MatrixIteratorAdapter<T, Itr>& operator+=(const long movement) {
            if (movement > 0) {
                for (long i = 0; i < movement; ++i) {
                    if (ip < max_ip) {
                        bool useStriding = ((ip + 1) < max_ip && ((ip + 1) % width == 0 || width == 1));
                        itr += (useStriding ? stride + 1 : 1);
                        ++ip;
                    }
                    else if (ip == max_ip) {
                        ++ip;
                        ++itr;
                    }
                }
                return *this;
            }
            return operator -= (std::abs(movement));
        }
        
        inline MatrixIteratorAdapter<T, Itr>& operator-=(const long movement) {
            if (movement > 0) {
                for (unsigned long i = 0; i < movement; ++i) {
                    if (ip > 0) {
                        bool useStriding = (ip != 0 && (ip % width == 0 || width == 1));
                        itr.operator -= (useStriding ? stride + 1 : 1);
                        --ip;
                    }
                }
                return *this;
            }
            return operator += (std::abs(movement));
        }
        
        inline MatrixIteratorAdapter<T, Itr>& operator++() {
            return operator +=(1);
        }
        inline MatrixIteratorAdapter<T, Itr>& operator--() {
            return operator -=(1);
        }
        
        inline MatrixIteratorAdapter<T, Itr> operator++(int) {
            auto temp(*this);
            operator ++();
            return temp;
        }
        inline MatrixIteratorAdapter<T, Itr> operator--(int) {
            auto temp(*this);
            operator --();
            return temp;
        }
        
        inline MatrixIteratorAdapter<T, Itr> operator+(const long movement) {
            auto temp(*this);
            temp += movement;
            return temp;
        }
        
        inline MatrixIteratorAdapter<T, Itr> operator-(const long movement) {
            auto temp(*this);
            temp -= movement;
            return temp;
        }
        
        inline ptrdiff_t operator-(const MatrixIteratorAdapter<T, Itr>& iterator) {
            return std::distance(iterator.getPtr(), this->getPtr());
        }
        
        inline T& operator*() {
            return itr.operator *();
        }
        inline const T& operator*() const {
            return itr.operator *();
        }
        inline T* operator->() {
            return itr.operator ->();
        }
        
        inline T* getPtr() const {
            return itr.getPtr();
        }
        inline const T* getConstPtr() const {
            return itr.getConstPtr();
        }
        
    private:
        inline MatrixIteratorAdapter<T, Itr>& operator=(T* ptr) { return *this; }
        
    protected:
        Itr itr;
        unsigned long ip;
        unsigned long max_ip;
        
        unsigned long width;
        unsigned long height;
        unsigned long stride;
    };
    
    template <typename T, class Itr> class MatrixReverseIteratorAdapter : public MatrixIteratorAdapter<T, Itr> {
    public:
        inline MatrixReverseIteratorAdapter(const Itr& origin, unsigned long width, unsigned long height, unsigned long stride) : MatrixIteratorAdapter<T, Itr>(origin, width, height, stride) {}
        inline MatrixReverseIteratorAdapter(const MatrixIteratorAdapter<T, Itr>& reverseIterator) = default;
        ~MatrixReverseIteratorAdapter() {}
        
        MatrixReverseIteratorAdapter<T, Itr>& operator=(const RawReverseIterator<T>& rawReverseIterator) = default;
        
        MatrixReverseIteratorAdapter<T, Itr>& operator+=(const long movement) {
            MatrixIteratorAdapter<T, Itr>::operator -= (movement);
            return *this;
        }
        MatrixReverseIteratorAdapter<T, Itr>& operator-=(const long movement) {
            MatrixIteratorAdapter<T, Itr>::operator += (movement);
            return *this;
        }
        MatrixReverseIteratorAdapter<T, Itr>& operator++() {
            MatrixIteratorAdapter<T, Itr>::operator -= (1);
            return *this;
        }
        MatrixReverseIteratorAdapter<T, Itr>& operator--() {
            MatrixIteratorAdapter<T, Itr>::operator += (1);
            return *this;
        }
        MatrixReverseIteratorAdapter<T, Itr> operator++(int) {
            auto temp(*this);
            MatrixIteratorAdapter<T, Itr>::operator --();
            return temp;
        }
        MatrixReverseIteratorAdapter<T, Itr> operator--(int) {
            auto temp(*this);
            MatrixIteratorAdapter<T, Itr>::operator ++();
            return temp;
        }
        MatrixReverseIteratorAdapter<T, Itr> operator+(const long movement) {
            auto temp(*this);
            temp -= movement;
            return temp;
        }
        MatrixReverseIteratorAdapter<T, Itr> operator-(const long movement) {
            auto temp(*this);
            temp += movement;
            return temp;
        }
        
        ptrdiff_t operator-(const MatrixReverseIteratorAdapter<T, Itr>& reverseIterator) {
            return std::distance(this->getPtr(), reverseIterator.getPtr());
        }
        
        MatrixIteratorAdapter<T, Itr> base() {
            MatrixIteratorAdapter<T, Itr> forwardIterator(&this);
            ++forwardIterator;
            return forwardIterator;
        }
    };
    
    typedef MatrixIteratorAdapter<double, std::vector<double>::iterator> matrix_iterator;
    typedef MatrixIteratorAdapter<const double, std::vector<const double>::const_iterator> const_matrix_iterator;
    
    typedef MatrixReverseIteratorAdapter<double, std::vector<double>::iterator> matrix_reverse_iterator;
    typedef MatrixReverseIteratorAdapter<const double, std::vector<const double>::const_iterator> const_matrix_reverse_iterator;
}}

#endif /* Iterators_h */
