//
//  InputVector.hpp
//  astra-nn
//
//  Created by Pavel on 23/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef InputVector_h
#define InputVector_h

#include "Vector.hpp"

namespace astra {
    
    class InputVector : public Vector {
    public:
        InputVector() {};
        explicit InputVector(unsigned long size) : Vector(size) {}
        explicit InputVector(const Vector& vec) : Vector(vec.size() + 1) {
            fill_from(vec);
        }
        
        InputVector& fill_from(const Vector& src) {
            std::vector<double>& storage = Vector::storage;
            unsigned long size = src.size() + 1;
            
            if (Vector::size() != size) {
                storage.resize(size);
            }
            
            std::copy(src.get_storage().begin(), src.get_storage().end(), storage.begin());
            storage[Vector::size() - 1] = 1;
            
            return *this;
        }
    };
    
    typedef InputVector inputv;
}


#endif /* InputVector_h */
