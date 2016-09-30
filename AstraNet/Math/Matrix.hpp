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
    
        unsigned long sum() const {
            unsigned long result = 0;
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
        
        friend Matrix operator*(const Matrix& left, const Matrix& right) {
            Matrix result(left.nCols, right.nRows);
            
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
        
        std::vector<Vector>& get_rows() {
            return rows;
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
        
        
    protected:
        
        std::vector<Vector> rows;
        unsigned long nCols, nRows;
    };
    
    typedef Matrix mat;
}

#endif /* Matrix_h */
