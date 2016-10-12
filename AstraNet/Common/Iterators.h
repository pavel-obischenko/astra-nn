//
//  Iterators.h
//  astra-nn
//
//  Created by Pavel on 11/10/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef Iterators_h
#define Iterators_h

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
    
    template <typename T> class StrideIteratorAdapter : public RawIterator<T> {
    public:
        explicit inline StrideIteratorAdapter(const RawIterator<T>& itr, unsigned long stride) : itr(itr), stride(stride) {}
        inline StrideIteratorAdapter(const StrideIteratorAdapter<T>& adapter) : stride(adapter.stride), itr(RawIterator<T>(adapter.getPtr())) {} // default ?
        ~StrideIteratorAdapter() {}
        
        inline StrideIteratorAdapter<T>& operator=(const StrideIteratorAdapter<T>& adapter) = default;
        
        inline operator bool() const {
            return itr.operator bool();
        }
        
        inline bool operator==(const StrideIteratorAdapter<T>& other) const {
            return itr.operator ==(other.itr);
        }
        inline bool operator!=(const StrideIteratorAdapter<T>& other) const {
            return itr.operator !=(other.itr);
        }
        
        inline StrideIteratorAdapter<T>& operator+=(const long movement) {
            itr.operator +=(movement * stride);
            return *this;
        }
        inline StrideIteratorAdapter<T>& operator-=(const long movement) {
            itr.operator -=(movement * stride);
            return *this;
        }
        
        inline StrideIteratorAdapter<T>& operator++() {
            return operator +=(1);
        }
        inline StrideIteratorAdapter<T>& operator--() {
            return operator -=(1);
        }
        
        inline StrideIteratorAdapter<T> operator++(int) {
            auto temp(*this);
            operator ++();
            return temp;
        }
        inline StrideIteratorAdapter<T> operator--(int) {
            auto temp(*this);
            operator --();
            return temp;
        }
        
        inline StrideIteratorAdapter<T> operator+(const long movement) {
            auto temp(*this);
            temp += movement;
            return temp;
        }
        
        inline StrideIteratorAdapter<T> operator-(const long movement) {
            auto temp(*this);
            temp -= movement;
            return temp;
        }
        
        inline ptrdiff_t operator-(const StrideIteratorAdapter<T>& strideIterator) {
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
        inline StrideIteratorAdapter<T>& operator=(T* ptr) { return *this; }
        
    protected:
        RawIterator<T> itr;
        unsigned long stride;
    };
    
    template <typename T> class RectIteratorAdapter : public RawIterator<T> {
    public:
        explicit inline RectIteratorAdapter(const RawIterator<T>& origin, unsigned long width, unsigned long height, unsigned long stride) : ip(0), max_ip(width * height), itr(origin), width(width), height(height), stride(stride) {}
        
        inline RectIteratorAdapter(const RectIteratorAdapter<T>& adapter) = default;
        ~RectIteratorAdapter() {}
        
        inline RectIteratorAdapter<T>& operator=(const RectIteratorAdapter<T>& adapter) = default;
        
        inline operator bool() const {
            return itr.operator bool();
        }
        
        inline bool operator==(const RectIteratorAdapter<T>& other) const {
            return itr.operator ==(other.itr);
        }
        inline bool operator!=(const RectIteratorAdapter<T>& other) const {
            return itr.operator !=(other.itr);
        }
        
        inline RectIteratorAdapter<T>& operator+=(const long movement) {
            if (movement > 0) {
                for (long i = 0; i < movement; ++i) {
                    if (ip < max_ip) {
                        itr.operator += (ip != 0 && (ip + 1) < max_ip && (ip + 1) % width == 0 ? stride + 1 : 1);
                        ++ip;
                    }
                }
                return *this;
            }
            else {
                return operator -= (std::abs(movement));
            }
        }
        inline RectIteratorAdapter<T>& operator-=(const long movement) {
            if (movement > 0) {
                for (unsigned long i = 0; i < movement; ++i) {
                    if (ip > 0) {
                        itr.operator -= (ip != 0 && ip % width == 0 ? stride + 1 : 1);
                        --ip;
                    }
                }
                
                return *this;
            }
            else {
                return operator += (std::abs(movement));
            }
        }
        
        inline RectIteratorAdapter<T>& operator++() {
            return operator +=(1);
        }
        inline RectIteratorAdapter<T>& operator--() {
            return operator -=(1);
        }
        
        inline RectIteratorAdapter<T> operator++(int) {
            auto temp(*this);
            operator ++();
            return temp;
        }
        inline RectIteratorAdapter<T> operator--(int) {
            auto temp(*this);
            operator --();
            return temp;
        }
        
        inline RectIteratorAdapter<T> operator+(const long movement) {
            auto temp(*this);
            temp += movement;
            return temp;
        }
        
        inline RectIteratorAdapter<T> operator-(const long movement) {
            auto temp(*this);
            temp -= movement;
            return temp;
        }
        
        inline ptrdiff_t operator-(const RectIteratorAdapter<T>& iterator) {
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
        inline RectIteratorAdapter<T>& operator=(T* ptr) { return *this; }
        
    protected:
        RawIterator<T> itr;
        unsigned long ip;
        unsigned long max_ip;
        
        unsigned long width;
        unsigned long height;
        unsigned long stride;
    };
    
    typedef RawIterator<double> iterator;
    typedef RawIterator<const double> const_iterator;
    
    typedef StrideIteratorAdapter<double> stride_iterator;
    typedef StrideIteratorAdapter<const double> const_stride_iterator;
    
    typedef RectIteratorAdapter<double> rect_iterator;
    typedef RectIteratorAdapter<const double> const_rect_iterator;
    
    typedef RawReverseIterator<double>       reverse_iterator;
    typedef RawReverseIterator<const double> const_reverse_iterator;
    
}}

#endif /* Iterators_h */
