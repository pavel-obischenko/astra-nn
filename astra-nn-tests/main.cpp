//
//  main.cpp
//  astra-nn-tests
//
//  Created by Pavel on 14/10/16.
//  Copyright (c) 2016 pavel. All rights reserved.
//

#include <iostream>
#include <gtest/gtest.h>

#include "../AstraNet/Math/Matrix.hpp"

class TestMatrix : public testing::Test
{
public:
    TestMatrix() : m3x3_0(3, 3), m3x3_1(3, 3), em3x3_0(3, 3) {}

protected:
    void SetUp() {
        m3x3_0 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        em3x3_0 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        m3x3_1 = {{7, 8, 9}, {4, 5, 6}, {1, 2, 3}};
    }
    void TearDown() {
    }

    astra::math::Matrix m3x3_0, m3x3_1, em3x3_0;
};

TEST_F(TestMatrix, equal_operator_test) {
    ASSERT_TRUE(m3x3_0 == em3x3_0);
    ASSERT_FALSE(m3x3_0 == m3x3_1);
}


int main(int argc, const char *argv[]) {
    testing::InitGoogleTest(&argc, const_cast<char**>(argv));
    return RUN_ALL_TESTS();
}
