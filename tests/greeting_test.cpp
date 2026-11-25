#include "greeting.h"

#include <gtest/gtest.h>

TEST(GreetingTest, ReturnsHelloWorld)
{
    EXPECT_STREQ(greeting(), "Hello, world!");
}
