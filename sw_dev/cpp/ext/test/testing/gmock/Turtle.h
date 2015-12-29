#if !defined(__TEST_FRAMEWORK__GMOCK__TURTLE_H_)
#define __TEST_FRAMEWORK__GMOCK__TURTLE_H_ 1


class Turtle
{
public:
    Turtle()  {}
    virtual ~Turtle()  {}

public:
    virtual void PenUp() = 0;
    virtual void PenDown() = 0;
    virtual void Forward(int distance) = 0;
    virtual void Turn(int degrees) = 0;
    virtual void GoTo(int x, int y) = 0;
    virtual int GetX() const = 0;
    virtual int GetY() const = 0;
};


#endif  // __TEST_FRAMEWORK__GMOCK__TURTLE_H_
