#include "TurtleMock.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>


namespace {
namespace local {

class Painter
{
public:
    Painter(Turtle *turtle)
    : turtle_(turtle)
    {}

public:
    bool DrawCircle(int cx, int cy, int radius)
    {
        if (turtle_)
        {
            turtle_->GoTo(cx, cy);
            turtle_->Forward(radius);
            turtle_->PenDown();
        }

        std::cout << "draw a circle of radius " << radius << " at (" << cx << ',' << cy << ")." << std::endl;

        if (turtle_)
        {
            turtle_->PenUp();
        }
    }

private:
    Turtle *turtle_;
};

}  // namespace local
}  // unnamed namespace

namespace my_gmock {

}  // namespace my_gmock

TEST(PainterTest, CanDrawSomething)
{
    const int cx = 0, cy = 0, radius = 10;

    TurtleMock turtle;

    EXPECT_CALL(turtle, PenUp()).Times(testing::AtLeast(1));
    EXPECT_CALL(turtle, PenDown()).Times(testing::AtLeast(1));
    EXPECT_CALL(turtle, Forward(radius / 2)).Times(testing::AtLeast(1));
    EXPECT_CALL(turtle, Turn(0)).Times(testing::AtLeast(1));
    EXPECT_CALL(turtle, GoTo(cx, cy)).Times(testing::AtLeast(1));

    local::Painter painter(&turtle);

    EXPECT_TRUE(painter.DrawCircle(cx, cy, radius));
}
