//
//  MatrixProxy.h
//  astra-nn
//
//  Created by Pavel on 12/10/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef MatrixProxy_h
#define MatrixProxy_h

#include "../Common/Iterators.h"
#include "Matrix.hpp"

namespace astra {
namespace math {

    class MatrixProxy : public Matrix {
        friend class astra::math::Matrix;
        
    protected:
        MatrixProxy() {}
        
    public:
        inline MatrixProxy(double* origin, unsigned long width, unsigned long height, unsigned long parentWidth) : beginItr(origin), stride(parentWidth - width) {
            set_width(width);
            set_height(height);
            
            endItr = beginItr.operator +(((width * height) + stride * (height - 1)));
        }
        MatrixProxy(const MatrixProxy& other) = default;
        
    public:
        friend std::ostream& operator<<(std::ostream& os, const MatrixProxy& mat) {
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
        
        inline MatrixProxy element_wise_mul(double arg) const {
            MatrixProxy result(*this);
            std::transform(begin(), end(), result.begin(), std::bind2nd(std::multiplies<double>(), arg));
            return result;
        }
        
        friend inline MatrixProxy& operator*=(MatrixProxy& left, double right) {
            std::transform(left.begin(), left.end(), left.begin(), std::bind2nd(std::multiplies<double>(), right));
            return left;
        }
        
        template <class _Function> inline _Function for_each(_Function __f) {
            return astra::math::for_each(this, __f);
        }
        
        template <class _Function> inline _Function for_each(_Function __f) const {
            return astra::math::for_each(this, __f);
        }
        
    public:
        inline common::rect_iterator begin() {
            return common::rect_iterator(beginItr, get_width(), get_height(), stride);
        }
        inline common::rect_iterator end() {
            return common::rect_iterator(endItr, get_width(), get_height(), stride);
        }
        
        inline common::const_rect_iterator begin() const {
            return common::const_rect_iterator(common::const_iterator(beginItr.getConstPtr()), get_width(), get_height(), stride);
        }
        inline common::const_rect_iterator end() const {
            return common::const_rect_iterator(common::const_iterator(endItr.getConstPtr()), get_width(), get_height(), stride);
        }
        
    private:
        common::iterator beginItr;
        common::iterator endItr;
        unsigned long stride;
    };
    
}}

#endif /* MatrixProxy_h */
