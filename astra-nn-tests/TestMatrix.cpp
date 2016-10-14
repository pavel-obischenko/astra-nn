//
// Created by Pavel on 14/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "../AstraNet/Math/Matrix.hpp"

#include <gtest/gtest.h>


class TestMatrix : public testing::Test
{
public:
    TestMatrix() : m3x3_0(3, 3), m3x3_1(3, 3), em3x3_0(3, 3), m1x3_0(3, 1), m3x1_0(1, 3) {}

protected:
    void SetUp() {
        m3x3_0 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        em3x3_0 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        m3x3_1 = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
        m1x3_0 = {{1, 2, 3}};
        m3x1_0 = {{1}, {2}, {3}};
    }
    void TearDown() {
    }

    astra::math::Matrix m3x3_0, m3x3_1, em3x3_0, m1x3_0, m3x1_0;
    //astra::math::Vector v9_0, v9_1;
};

TEST_F(TestMatrix, test_equal_operator) {
    ASSERT_TRUE(m3x3_0 == em3x3_0);
    ASSERT_FALSE(m3x3_0 == m3x3_1);
}

TEST_F(TestMatrix, test_add_operators) {
    astra::math::Matrix res0 = {{10, 10, 10}, {10, 10, 10}, {10, 10, 10}};
    astra::math::Matrix res1 = {{2, 4, 6}};
    astra::math::Matrix res2 = {{2}, {4}, {6}};

    ASSERT_EQ(m3x3_0 + m3x3_1, res0);
    ASSERT_EQ(m1x3_0 + m1x3_0, res1);
    ASSERT_EQ(m3x1_0 + m3x1_0, res2);
}

TEST_F(TestMatrix, test_dot_product) {
    ASSERT_DOUBLE_EQ(m3x3_0.dot_product(m3x3_0), 285.0);
}