//
// Created by Pavel on 14/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "../AstraNet/Math/Matrix.h"
#include "../AstraNet/Math/Vector.h"

#include <gtest/gtest.h>

using namespace astra::math;

class TestMath : public testing::Test
{
public:
    TestMath() : m3x3_0(3, 3), m3x3_1(3, 3), em3x3_0(3, 3), m1x3_0(3, 1), m3x1_0(1, 3) {}

protected:
    void SetUp() {
        m3x3_0 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        em3x3_0 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        m3x3_1 = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
        m1x3_0 = {{1, 2, 3}};
        m3x1_0 = {{1}, {2}, {3}};

        v3_0 = {1, 2, 3};
        v3_1 = {4, 5, 6};
    }
    void TearDown() {
    }

    Matrix m3x3_0, m3x3_1, em3x3_0, m1x3_0, m3x1_0;
    Vector v3_0, v3_1;
};

TEST_F(TestMath, test_equal_operator) {
    ASSERT_TRUE(m3x3_0 == em3x3_0);
    ASSERT_FALSE(m3x3_0 == m3x3_1);
}

TEST_F(TestMath, test_dot_product) {
    ASSERT_DOUBLE_EQ(m3x3_0.dot_product(m3x3_0), 285.0);
}

TEST_F(TestMath, test_element_wise_mul) {
    Matrix res = {{2, 4, 6}, {8, 10, 12}, {14, 16, 18}};
    ASSERT_EQ(m3x3_0.element_wise_mul(2), res);
}

TEST_F(TestMath, test_add_operators) {
    Matrix res0 = {{10, 10, 10}, {10, 10, 10}, {10, 10, 10}};
    Matrix res1 = {{2, 4, 6}};
    Matrix res2 = {{2}, {4}, {6}};

    ASSERT_EQ(m3x3_0 + m3x3_1, res0);
    ASSERT_EQ(m1x3_0 + m1x3_0, res1);
    ASSERT_EQ(m3x1_0 + m3x1_0, res2);
    ASSERT_EQ(m3x3_0 += m3x3_1, res0);
}

TEST_F(TestMath, test_sub_operators) {
    Matrix res0 = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    Matrix res1 = {{0, 0, 0}};
    Matrix res2 = {{0}, {0}, {0}};

    ASSERT_EQ(m3x3_0 - m3x3_0, res0);
    ASSERT_EQ(m1x3_0 - m1x3_0, res1);
    ASSERT_EQ(m3x1_0 - m3x1_0, res2);
}

TEST_F(TestMath, test_mul_operators) {
    Matrix res0 = {{30, 24, 18}, {84, 69, 54}, {138, 114, 90}};
    Matrix res1 = {{14}, {32}, {50}};
    Matrix res2 = {{30, 36, 42}};
    Matrix res3 = {{2, 4, 6}, {8, 10, 12}, {14, 16, 18}};

    ASSERT_EQ(m3x3_0 * m3x3_1, res0);
    ASSERT_EQ(m3x3_0 * m3x1_0, res1);
    ASSERT_EQ(m1x3_0 * m3x3_0, res2);
    ASSERT_EQ(m3x3_0 * 2, res3);
    ASSERT_EQ(2 * m3x3_0, res3);
    ASSERT_EQ(m3x3_0 *= 2, res3);
}

TEST_F(TestMath, test_transpose) {
    Matrix res0 = {{1, 4, 7}, {2, 5, 8}, {3, 6, 9}};
    Matrix res1 = {{1}, {2}, {3}};
    Matrix res2 = {{1, 2, 3}};

    ASSERT_EQ(m3x3_0.transpose(), res0);
    ASSERT_EQ(m1x3_0.transpose(), res1);
    ASSERT_EQ(m3x1_0.transpose(), res2);
}

TEST_F(TestMath, test_vector_multiplications) {
    Matrix res0 = {{4, 5, 6}, {8, 10, 12}, {12, 15, 18}};
    Matrix res1 = {{1, 2, 3}, {2, 4, 6}, {3, 6, 9}};

    ASSERT_EQ(v3_0 * v3_1, res0);
    ASSERT_EQ(v3_0 * v3_0, res1);
}

TEST_F(TestMath, test_matrix_to_vector_multiplications) {
    Vector res1 = {14, 32, 50};
    ASSERT_EQ(m3x3_0 * v3_0, res1);
}

TEST_F(TestMath, test_matrix_initialization) {
    Matrix res0 = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    Matrix res1 = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    Matrix m0 = Matrix::zero(3, 3);
    ASSERT_EQ(m0, res0);

    Matrix m1 = Matrix::identity(3);
    ASSERT_EQ(m1, res1);
}