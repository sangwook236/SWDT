#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/SE3StateSpace.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/SimpleSetup.h>
#include <iostream>


namespace {
namespace local {

bool isStateValid(const ompl::base::State *state)
{
    // cast the abstract state type to the type we expect
    const ompl::base::SE3StateSpace::StateType *se3state = state->as<ompl::base::SE3StateSpace::StateType>();

    // extract the first component of the state and cast it to what we expect
    const ompl::base::RealVectorStateSpace::StateType *pos = se3state->as<ompl::base::RealVectorStateSpace::StateType>(0);

    // extract the second component of the state and cast it to what we expect
    const ompl::base::SO3StateSpace::StateType *rot = se3state->as<ompl::base::SO3StateSpace::StateType>(1);

    // check validity of state defined by pos & rot


    // return a value that is always true but uses the two variables we define, so we avoid compiler warnings
    return (const void*)rot != (const void*)pos;
}

void planWithoutSimpleSetup()
{
    // construct the state space we are planning in
    ompl::base::StateSpacePtr space(new ompl::base::SE3StateSpace());

    // set the bounds for the R^3 part of SE(3)
    ompl::base::RealVectorBounds bounds(3);
    bounds.setLow(-1);
    bounds.setHigh(1);

    space->as<ompl::base::SE3StateSpace>()->setBounds(bounds);

    // construct an instance of  space information from this state space
    ompl::base::SpaceInformationPtr si(new ompl::base::SpaceInformation(space));

    // set state validity checking for this space
    si->setStateValidityChecker(boost::bind(&isStateValid, _1));

    // create a random start state
    ompl::base::ScopedState<> start(space);
    start.random();

    // create a random goal state
    ompl::base::ScopedState<> goal(space);
    goal.random();

    // create a problem instance
    ompl::base::ProblemDefinitionPtr pdef(new ompl::base::ProblemDefinition(si));

    // set the start and goal states
    pdef->setStartAndGoalStates(start, goal);

    // create a planner for the defined space
    ompl::base::PlannerPtr planner(new ompl::geometric::RRTConnect(si));

    // set the problem we are trying to solve for the planner
    planner->setProblemDefinition(pdef);

    // perform setup steps for the planner
    planner->setup();


    // print the settings for this space
    si->printSettings(std::cout);

    // print the problem settings
    pdef->print(std::cout);

    // attempt to solve the problem within one second of planning time
    ompl::base::PlannerStatus solved = planner->solve(1.0);

    if (solved)
    {
        // get the goal representation from the problem definition (not the same as the goal state)
        // and inquire about the found path
        ompl::base::PathPtr path = pdef->getSolutionPath();
        std::cout << "Found solution:" << std::endl;

        // print the path to screen
        path->print(std::cout);
    }
    else
        std::cout << "No solution found" << std::endl;
}

void planWithSimpleSetup()
{
    // construct the state space we are planning in
    ompl::base::StateSpacePtr space(new ompl::base::SE3StateSpace());

    // set the bounds for the R^3 part of SE(3)
    ompl::base::RealVectorBounds bounds(3);
    bounds.setLow(-1);
    bounds.setHigh(1);

    space->as<ompl::base::SE3StateSpace>()->setBounds(bounds);

    // define a simple setup class
    ompl::geometric::SimpleSetup ss(space);

    // set state validity checking for this space
    ss.setStateValidityChecker(boost::bind(&isStateValid, _1));

    // create a random start state
    ompl::base::ScopedState<> start(space);
    start.random();

    // create a random goal state
    ompl::base::ScopedState<> goal(space);
    goal.random();

    // set the start and goal states
    ss.setStartAndGoalStates(start, goal);

    // this call is optional, but we put it in to get more output information
    ss.setup();
    ss.print();

    // attempt to solve the problem within one second of planning time
    ompl::base::PlannerStatus solved = ss.solve(1.0);

    if (solved)
    {
        std::cout << "Found solution:" << std::endl;
        // print the path to screen
        ss.simplifySolution();
        ss.getSolutionPath().print(std::cout);
    }
    else
        std::cout << "No solution found" << std::endl;
}
	
}  // namespace local
}  // unnamed namespace

namespace my_ompl {

// [ref] ${OMPL_HOME}/demos/RigidBodyPlanning.cpp
void rigid_body_planning_example()
{
	//local::planWithoutSimpleSetup();

	local::planWithSimpleSetup();
}

}  // namespace my_ompl
