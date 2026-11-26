#include "frame_budget.h"

#include <gtest/gtest.h>

namespace {
TEST(FrameBudget, SelectsSustainableTierWithPromotionHysteresis)
{
    haggis::FrameBudget budget;
    EXPECT_EQ(budget.target_fps(), 120);
    budget.update_target(4.2, 8.4, 1);
    EXPECT_EQ(budget.target_fps(), 120);
    budget.update_target(4.2, 8.4, 1);
    EXPECT_EQ(budget.target_fps(), 240);
    budget.update_target(8.4, 8.4, 1);
    EXPECT_EQ(budget.target_fps(), 120);
    budget.update_target(16.7, 16.7, 1);
    EXPECT_EQ(budget.target_fps(), 60);
}

TEST(FrameBudget, AcceptsUnevenRefreshIntervalsOn144HzDisplay)
{
    haggis::FrameBudget budget;
    const double refresh = 1000.0 / 144.0;
    budget.update_target(refresh, 2 * refresh, 1);
    EXPECT_EQ(budget.target_fps(), 120);
}

TEST(FrameBudget, AccruesTenPercentAndBoundsDispatchGrowth)
{
    haggis::FrameBudget budget;
    EXPECT_EQ(budget.workgroups_for(1), 0u);
    budget.add_frame_credit(0.4);
    EXPECT_EQ(budget.workgroups_for(1), 0u);
    budget.add_frame_credit(0.5);
    EXPECT_EQ(budget.workgroups_for(1), 1u);
    budget.spend_credit(0.08);
    budget.observe_cost(1, 0.08);
    budget.add_frame_credit(1000);
    EXPECT_EQ(budget.workgroups_for(1), 2u);
}

TEST(FrameBudget, LendsSpareTimeOnlyWhenKeepingPace)
{
    haggis::FrameBudget budget;
    budget.update_target(8.3, 8.4, 1);
    EXPECT_EQ(budget.workgroups_for(1), 1u);
    budget.spend_credit(0.08);
    budget.observe_cost(1, 0.08);
    budget.update_target(8.3, 22, 1);
    EXPECT_EQ(budget.workgroups_for(1), 0u);
}
TEST(FrameBudget, MinimumProgressSurvivesAnExpensiveWorkgroupAndBusyRenderer)
{
    haggis::FrameBudget budget;
    budget.update_target(16.7, 16.7, 18);
    budget.add_frame_credit(budget.period());
    ASSERT_EQ(budget.workgroups_for(18), 1u);
    budget.spend_credit(40.0);
    budget.observe_cost(1, 40.0);
    for (int i = 0; i < 10; ++i) budget.add_frame_credit(budget.period());
    EXPECT_EQ(budget.workgroups_for(18), 1u);
}

TEST(FrameBudget, DeliversReservedShareAcrossFramesWithIndivisibleDispatches)
{
    haggis::FrameBudget budget;
    double spent = 0;
    constexpr double group_ms = 0.36;
    for (int frame = 0; frame < 240; ++frame) {
        budget.add_frame_credit(budget.period());
        const auto groups = budget.workgroups_for(5.0);
        const auto cost = groups * group_ms;
        budget.spend_credit(cost);
        budget.observe_cost(groups, cost);
        spent += cost;
    }
    EXPECT_NEAR(spent, 0.1 * 240 * budget.period(), 2 * group_ms);
}
} // namespace
