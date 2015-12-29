#if !defined(__TEST_FRAMEWORK__GMOCK__TURTLE_MOCK_H_)
#define __TEST_FRAMEWORK__GMOCK__TURTLE_MOCK_H_ 1


#include "Turtle.h"
#include <gmock/gmock.h>


class TurtleMock : public Turtle
{
public:
    MOCK_METHOD0(PenUp, void());
    MOCK_METHOD0(PenDown, void());
    MOCK_METHOD1(Forward, void(int distance));
    MOCK_METHOD1(Turn, void(int degrees));
    MOCK_METHOD2(GoTo, void(int x, int y));
    MOCK_CONST_METHOD0(GetX, int());
    MOCK_CONST_METHOD0(GetY, int());
};


#endif  // __TEST_FRAMEWORK__GMOCK__TURTLE_MOCK_H_
