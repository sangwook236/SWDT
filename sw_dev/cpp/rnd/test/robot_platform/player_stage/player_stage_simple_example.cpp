//#include "stdafx.h"
#include <libplayerc++/playerc++.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_player_stage {

void simple_example()
{
	PlayerCc::PlayerClient robot("localhost");

	PlayerCc::Position2dProxy positionProxy(&robot, 0);
	PlayerCc::SonarProxy sonarProxy(&robot, 0);

	for (;;)
	{
		// read from the proxies
		robot.Read();

		// print out sonars for fun
		std::cout << sonarProxy << std::endl;

		// do simple collision avoidance
		double turnrate;
		if ((sonarProxy[0] + sonarProxy[1]) < (sonarProxy[6] + sonarProxy[7]))
			turnrate = PlayerCc::dtor(-20);  // turn 20 degrees per second
		else
			turnrate = PlayerCc::dtor(20);

		double speed;
		if (sonarProxy[3] < 0.500)
			speed = 0;
		else
			speed = 0.100;

		// command the motors
		positionProxy.SetSpeed(speed, turnrate);
	}
}

}  // namespace my_player_stage
