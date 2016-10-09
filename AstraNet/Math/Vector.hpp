//
//  Vector.hpp
//  astra-nn
//
//  Created by Pavel on 21/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef Vector_hpp
#define Vector_hpp

#include <vector>
#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>

namespace astra {
    
    class Vector {
        friend class Matrix;
    public:
        Vector() {};
        Vector(const Vector& vector) : storage(vector.storage) {}
        Vector(const std::initializer_list<double>& init) : storage(init) {}
        explicit Vector(unsigned long size) : storage(size) {}
        explicit Vector(const std::vector<double>& vec) : storage(vec) {}
        
        unsigned long size() const {
            return storage.size();
        }
        
        double sum() const {
            double result = 0;
            std::for_each(storage.begin(), storage.end(), [&result](double val) {
                result += val;
            });
            return result;
        }
        
        Vector mul_termwise(const Vector& arg) const {
            if (arg.size() != size()) {
                // TODO: throw exception
            }
            
            std::vector<double> result(size());
            std::transform(storage.begin(), storage.end(), arg.storage.begin(), result.begin(), std::multiplies<double>());

            return Vector(result);
        }
        
        friend const Vector operator+(const Vector& left, const Vector& right) {
            if (!left.size() || left.size() != right.size()) {
                // TODO: throw exception
            }
            
            std::vector<double> result(left.size());
            std::transform(left.storage.begin(), left.storage.end(), right.storage.begin(), result.begin(), std::plus<double>());
            
            return Vector(result);
        }
        
        friend Vector& operator+=(Vector& left, const Vector& right) {
            left = left + right;
            return left;
        }
        
        friend Vector operator-(const Vector& left, const Vector& right) {
            if (!left.size() || left.size() != right.size()) {
                // TODO: throw exception
            }
            
            std::vector<double> result(left.size());
            std::transform(left.storage.begin(), left.storage.end(), right.storage.begin(), result.begin(), std::minus<double>());
            
            return Vector(result);
        }
        
        friend Vector& operator-=(Vector& left, const Vector& right) {
            left = left - right;
            return left;
        }
        
        friend const Vector operator*(const Vector& left, double right) {
            std::vector<double> result;
            
            std::for_each(left.storage.begin(), left.storage.end(), [&result, right](double val) {
                result.push_back(val * right);
            });
            
            return Vector(result);
        }
        
        friend const Vector operator*(double left, const Vector& right) {
            std::vector<double> result;
            
            std::for_each(right.storage.begin(), right.storage.end(), [&result, left](double val) {
                result.push_back(val * left);
            });
            
            return Vector(result);
        }
        
        friend const Vector operator*=(Vector& left, double right) {
            left = left * right;
            return left;
        }
        
        friend std::ostream& operator<<(std::ostream& os, const Vector& vec) {
            os << "{";
            for(auto item = vec.storage.begin(); item != vec.storage.end(); item++) {
                os << *item << (item != vec.storage.end() - 1 ? ", " : "");
            }
            os << "}";
            return os;
        }
        
        double& operator[](unsigned long index) {
            return storage[index];
        }
        
        const std::string to_string() const {
            std::stringstream strm;
            strm << this;
            return strm.str();
        }
        
        std::vector<double>& get_storage() {
            return storage;
        }
        
        const std::vector<double>& get_storage() const {
            return storage;
        }
        
    protected:
        std::vector<double> storage;
        
    };
    
    typedef Vector vec;
    typedef std::shared_ptr<Vector> VectorPtr;
}


#endif /* Vector_hpp */
