#include "frame_budget.h"

#include <gtest/gtest.h>

namespace {
TEST(FrameBudget, SelectsSustainableTierWithPromotionHysteresis)
{
    haggis::FrameBudget budget;
    EXPECT_EQ(budget.target_fps(), 120);
    budget.tune(4.2, 8.4, 1);
    EXPECT_EQ(budget.target_fps(), 120);
    budget.tune(4.2, 8.4, 1);
    EXPECT_EQ(budget.target_fps(), 240);
    budget.tune(8.4, 8.4, 1);
    EXPECT_EQ(budget.target_fps(), 120);
    budget.tune(16.7, 16.7, 1);
    EXPECT_EQ(budget.target_fps(), 60);
}

TEST(FrameBudget, AcceptsUnevenRefreshIntervalsOn144HzDisplay)
{
    haggis::FrameBudget budget;
    const double refresh = 1000.0 / 144.0;
    budget.tune(refresh, 2 * refresh, 1);
    EXPECT_EQ(budget.target_fps(), 120);
}

TEST(FrameBudget, AccruesTenPercentAndBoundsDispatchGrowth)
{
    haggis::FrameBudget budget;
    EXPECT_EQ(budget.take_chunk(1), 0u);
    budget.accrue(0.4);
    EXPECT_EQ(budget.take_chunk(1), 0u);
    budget.accrue(0.5);
    EXPECT_EQ(budget.take_chunk(1), 1u);
    budget.completed(1, 0.08);
    budget.accrue(1000);
    EXPECT_EQ(budget.take_chunk(1), 2u);
}

TEST(FrameBudget, LendsSpareTimeOnlyWhenKeepingPace)
{
    haggis::FrameBudget budget;
    budget.tune(8.3, 8.4, 1);
    EXPECT_EQ(budget.take_chunk(1), 1u);
    budget.completed(1, 0.08);
    budget.tune(8.3, 22, 1);
    EXPECT_EQ(budget.take_chunk(1), 0u);
}
TEST(FrameBudget, MinimumProgressSurvivesAnExpensiveWorkgroupAndBusyRenderer)
{
    haggis::FrameBudget budget;
    budget.tune(16.7, 16.7, 18);
    budget.accrue(budget.period());
    ASSERT_EQ(budget.take_chunk(18), 1u);
    budget.completed(1, 40.0);
    for (int i = 0; i < 10; ++i) budget.accrue(budget.period());
    EXPECT_EQ(budget.take_chunk(18), 1u);
}

TEST(FrameBudget, DeliversReservedShareAcrossFramesWithIndivisibleDispatches)
{
    haggis::FrameBudget budget;
    double spent = 0;
    constexpr double group_ms = 0.36;
    for (int frame = 0; frame < 240; ++frame) {
        budget.accrue(budget.period());
        const auto groups = budget.take_chunk(5.0);
        const auto cost = groups * group_ms;
        budget.completed(groups, cost);
        spent += cost;
    }
    EXPECT_NEAR(spent, 0.1 * 240 * budget.period(), 2 * group_ms);
}
} // namespace
