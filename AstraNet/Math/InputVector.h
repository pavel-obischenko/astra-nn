//
//  InputVector.hpp
//  astra-nn
//
//  Created by Pavel on 23/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef InputVector_h
#define InputVector_h

#include "Vector.h"

namespace astra {
namespace math {

    class InputVector : public Vector {
    public:
        InputVector() {}
        explicit InputVector(unsigned long size) : Vector(size) {}
        InputVector(const Vector& other) : Vector(other.size() + 1) {
            std::copy(other.begin(), other.end(), begin());
            operator[](Vector::size() - 1) = 1;
        }

        Vector toVector() const {
            return Vector(*this);
        }
    };
}}

#endif /* InputVector_h */
