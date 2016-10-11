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
        
        inline RawIterator<T>& operator+=(const ptrdiff_t& movement) {
            dataPtr += movement;
            return *this;
        }
        inline RawIterator<T>& operator-=(const ptrdiff_t& movement) {
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
        
        inline RawIterator<T> operator+(const ptrdiff_t& movement) {
            auto oldPtr = dataPtr;
            dataPtr += movement;
            auto temp(*this);
            dataPtr = oldPtr;
            return temp;
        }
        
        inline RawIterator<T> operator-(const ptrdiff_t& movement) {
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
        inline const T* getConstPtr() const {
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
        
        RawReverseIterator<T>& operator+=(const ptrdiff_t& movement) {
            this->dataPtr -= movement;
            return *this;
        }
        RawReverseIterator<T>& operator-=(const ptrdiff_t& movement) {
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
        RawReverseIterator<T> operator+(const int& movement) {
            auto oldPtr = this->dataPtr;
            this->dataPtr -= movement;
            auto temp(*this);
            this->dataPtr = oldPtr;
            return temp;
        }
        RawReverseIterator<T> operator-(const int& movement) {
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
        explicit inline StrideIteratorAdapter(RawIterator<T>* itr, unsigned long stride) : itr(itr), stride(stride) {}
        inline StrideIteratorAdapter(const StrideIteratorAdapter<T, Itr>& adapter) : stride(adapter.stride) {
            itr = new Itr(adapter.getPtr());
        };
        ~StrideIteratorAdapter() {}
        
        inline StrideIteratorAdapter<T, Itr>& operator=(const StrideIteratorAdapter<T, Itr>& adapter) = default;
        
        inline operator bool() const {
            return itr->operator bool();
        }
        
        inline bool operator==(const RawIterator<T>& rawIterator) const {
            return itr->operator ==(rawIterator);
        }
        inline bool operator!=(const RawIterator<T>& rawIterator) const {
            return itr->operator ==(rawIterator);
        }
        
        inline StrideIteratorAdapter<T, Itr>& operator+=(const ptrdiff_t& movement) {
            itr->operator+= (movement * stride);
            return *this;
        }
        inline StrideIteratorAdapter<T, Itr>& operator-=(const ptrdiff_t& movement) {
            itr->operator-= (movement * stride);
            return *this;
        }
        
        inline StrideIteratorAdapter<T, Itr>& operator++() {
            return operator+=(1);
        }
        inline StrideIteratorAdapter<T, Itr>& operator--() {
            return operator-=(1);
        }
        
        inline StrideIteratorAdapter<T, Itr> operator++(int) {
            auto temp(*this);
            operator++();
            return temp;
        }
        inline StrideIteratorAdapter<T, Itr> operator--(int) {
            auto temp(*this);
            operator--();
            return temp;
        }
        
        inline StrideIteratorAdapter<T, Itr> operator+(const int& movement) {
            auto temp(*this);
            temp += movement;
            return temp;
        }
        
        inline StrideIteratorAdapter<T, Itr> operator-(const int& movement) {
            auto temp(*this);
            temp -= movement;
            return temp;
        }
        
        inline ptrdiff_t operator-(const StrideIteratorAdapter<T, Itr>& strideIterator) {
            return std::distance(strideIterator.getPtr(), this->getPtr());
        }
        
        inline T& operator*() {
            return itr->operator*();
        }
        inline const T& operator*() const {
            return itr->operator*();
        }
        inline T* operator->() {
            return itr->operator->();
        }
        
        inline T* getPtr() const {
            return itr->getPtr();
        }
        inline const T* getConstPtr() const {
            return itr->getConstPtr();
        }
        
    private:
        inline StrideIteratorAdapter<T, Itr>& operator=(T* ptr) { return *this; }
        
    protected:
        Itr* itr;
        unsigned long stride;
    };
    
    typedef RawIterator<double> iterator;
    typedef RawIterator<const double> const_iterator;
    
    typedef RawReverseIterator<double>       reverse_iterator;
    typedef RawReverseIterator<const double> const_reverse_iterator;
    
}}

#endif /* Iterators_h */
