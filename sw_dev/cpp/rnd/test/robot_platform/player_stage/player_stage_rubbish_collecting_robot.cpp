//#include "stdafx.h"
#include <libplayerc++/playerc++.h>
#include <iostream>
#include <limits>
#include <cstdlib>
#include <cmath>


namespace {
namespace local {

struct item_t
{
	char name[16];
	double x;
	double y;
};

void wander(double &forwardSpeed, double &turnSpeed)
{
	const int maxSpeed = 1;
	const int maxTurn = 90;
	double fspeed, tspeed;

	// fspeed is between 0 and 10
	fspeed = rand() % 10;
	fspeed = (fspeed / 10.0) * maxSpeed;

	tspeed = rand() % (2 * maxTurn);
	tspeed -= maxTurn;

	forwardSpeed = fspeed;
	turnSpeed = tspeed;
}

void avoidObstacles(double &forwardSpeed, double &turnSpeed, PlayerCc::SonarProxy &sp)
{
	// will avoid obstacles colser than 40cm
	const double avoidDistance = 0.4;
	// will turn away at 60 degrees/ssec
	const int avoidTurnSpeed = 60;

	// left corner is sonar no. 2
	// right corner is sonar no. 3
	if (sp[2] < avoidDistance)
	{
		forwardSpeed = 0;
		// turn right
		turnSpeed = (-1) * avoidTurnSpeed;
	}
	else if (sp[3] < avoidDistance)
	{
		forwardSpeed = 0;
		// turn left
		turnSpeed = avoidTurnSpeed;
	}
	else if (sp[0] < avoidDistance && sp[1] < avoidDistance)
	{
		// back off a little bit
		forwardSpeed = -0.2;
		turnSpeed = avoidTurnSpeed;
	}
}

void moveToItem(double &forwardSpeed, double &turnSpeed, PlayerCc::BlobfinderProxy &bfp)
{
	const int noBlobs = bfp.GetCount();
	const int turningSpeed = 10;

	// numer of pixels away from the image center
	const int margin = 10;

	uint32_t biggestBlobArea = 0;
	int biggestBlob = 0;

	// find the largest blob
	for (int i = 0; i < noBlobs; ++i)
	{
		// get blob from proxy
		const playerc_blobfinder_blob_t currBlob = bfp[i];

		//if (currBlob.area > biggestBlobArea)
		if (currBlob.x * currBlob.y > biggestBlobArea)
		{
			biggestBlob = i;
			//biggestBlobArea = currBlob.area;
			biggestBlobArea = currBlob.x * currBlob.y;
		}
	}
	const playerc_blobfinder_blob_t blob = bfp[biggestBlob];

	// find center of image
	const int center = bfp.GetWidth() / 2;

	// adjust turn to center the blob in image
	// if the blob's center is within some margin of the image center then move forwards, otherwise turn so that it is centred
	if (blob.x < center - margin)  // blob to the left of center
	{
		forwardSpeed = 0;
		turnSpeed = turningSpeed;  // turn left
	}
	else if (blob.x > center + margin)  // blob to the right of center
	{
		forwardSpeed = 0;
		turnSpeed = -turningSpeed;  // turn right
	}
	else  // otherwise go straight ahead
	{
		forwardSpeed = 0.5;
		turnSpeed = 0;
	}
}

int findItem(const char *robotName, item_t *itemList, const int listLength, PlayerCc::SimulationProxy &sim)
{
	const double radius = 0.375;
	const double distBotToCircle = 0.625;

	// find the robot...
	double robotX, robotY, robotYaw;
	sim.GetPose2d((char *)robotName, robotX, robotY, robotYaw);

	// now we find the centre of the search circle. this is distBotToCircle metres from the robot's origin along its yaw.
	double circleX = distBotToCircle * std::cos(robotYaw);  // horizontal offset from robot origin
	double circleY = distBotToCircle * std::sin(robotYaw);  // vertical offset from robot origin

	// find actual centre relative to simulation.
	circleX = robotX + circleX;
	circleY = robotY + circleY;

	// to find which items are within this circle we find their Euclidian distance to the circle centre.
	// find the closest one and if it's distance is smaller than the circle radius then return its index.
	double smallestDist = std::numeric_limits<double>::max();
	int closestItem = 0;
	for (int i = 0; i < listLength; ++i)
	{
		const double x = circleX - itemList[i].x;
		const double y = circleY - itemList[i].y;

		// find Euclidian distance
		const double dist = std::sqrt((x*x) + (y*y));
		if (dist < smallestDist)
		{
			smallestDist = dist;
			closestItem = i;
		}
	}

	return closestItem;
}

void refreshItemList(item_t *itemList, PlayerCc::SimulationProxy &simProxy)
{
	// get the poses of the oranges
	for (int i = 0, k = 1; i < 4; ++i, ++k)
	{
		const char orangeStr[] = "orange%d";
		sprintf(itemList[i].name, orangeStr, k);

		double dummy;  // dummy variable, don't need yaws.
		simProxy.GetPose2d(itemList[i].name, itemList[i].x, itemList[i].y, dummy);
	}

	// get the poses of the cartons
	for (int i = 4, k = 1; i < 8; ++i, ++k)
	{
		const char cartonStr[] = "carton%d";
		sprintf(itemList[i].name, cartonStr, k);

		double dummy;  // dummy variable, don't need yaws.
		simProxy.GetPose2d(itemList[i].name, itemList[i].x, itemList[i].y, dummy);
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_player_stage {

#define __SIMULATION_MODE 0

void rubbish_collecting_robot(int argc, char *argv[])
{
	const char *robotNames[] = { "bob1", "bob2" };

	double forwardSpeed, turnSpeed;
	const size_t itemCount = 8;
	local::item_t itemList[itemCount];
	const double laserMaxDetectionRange = 0.25;

	//
	// first robot
	PlayerCc::PlayerClient robot("utopia.kaist.ac.kr", 6665);

	PlayerCc::SimulationProxy simProxy(&robot, 0);

	PlayerCc::Position2dProxy posProxy(&robot, 0);
	PlayerCc::SonarProxy sonarProxy(&robot, 0);
	PlayerCc::BlobfinderProxy blobProxy(&robot, 0);
	PlayerCc::LaserProxy laserProxy(&robot, 0);

	// second robot
#if __SIMULATION_MODE == 1
	PlayerCc::Position2dProxy posProxy2(&robot, 1);
	PlayerCc::SonarProxy sonarProxy2(&robot, 1);
	PlayerCc::BlobfinderProxy blobProxy2(&robot, 1);
	PlayerCc::LaserProxy laserProxy2(&robot, 1);
#elif __SIMULATION_MODE == 2
	PlayerCc::PlayerClient robot2("utopia.kaist.ac.kr", 6666);

	PlayerCc::SimulationProxy simProxy2(&robot2, 0);

	PlayerCc::Position2dProxy posProxy2(&robot2, 0);
	PlayerCc::SonarProxy sonarProxy2(&robot2, 0);
	PlayerCc::BlobfinderProxy blobProxy2(&robot2, 0);
	PlayerCc::LaserProxy laserProxy2(&robot2, 0);
#endif

	local::refreshItemList(itemList, simProxy);
#if __SIMULATION_MODE == 2
	refreshItemList(itemList, simProxy2);
#endif

	// enable motors
	posProxy.SetMotorEnable(1);
#if __SIMULATION_MODE == 1 || __SIMULATION_MODE == 2
	posProxy.SetMotorEnable(1);
#endif

	// request geometries
	posProxy.RequestGeom();
	sonarProxy.RequestGeom();
	laserProxy.RequestGeom();
	//blobProxy.RequestGeom();  // blobfinder doesn't have geometry
#if __SIMULATION_MODE == 1 || __SIMULATION_MODE == 2
	posProxy2.RequestGeom();
	sonarProxy2.RequestGeom();
	laserProxy2.RequestGeom();
	//blobProxy2.RequestGeom();  // blobfinder doesn't have geometry
#endif

	// here so that laserProxy[90] doesn't segfault on first loop
	robot.Read();
#if __SIMULATION_MODE == 2
	robot2.Read();
#endif

	while (true)
	{
		// read from the proxies
		robot.Read();
#if __SIMULATION_MODE == 2
		robot2.Read();
#endif

		if (blobProxy.GetCount() == 0)
		{
			// wander
			std::cout << "wandering" << std::endl;
			local::wander(forwardSpeed, turnSpeed);
		}
		else
		{
			// move towards the item
			std::cout << "moving to item" << std::endl;
			local::moveToItem(forwardSpeed, turnSpeed, blobProxy);

			// FIXME [delete] >>
			std::cout << "forwardSpeed: " << forwardSpeed << ", turnSpeed: " << turnSpeed << std::endl;
			double x, y, yaw;
			simProxy.GetPose2d("bob1", x, y, yaw);
			std::cout << "<robot> x: " << x << ", y: " << y << ", yaw: " << yaw << std::endl;
			simProxy.GetPose2d("orange1", x, y, yaw);
			std::cout << "<orange1> x: " << x << ", y: " << y << ", yaw: " << yaw << std::endl;
			simProxy.GetPose2d("orange2", x, y, yaw);
			std::cout << "<orange2> x: " << x << ", y: " << y << ", yaw: " << yaw << std::endl;
			simProxy.GetPose2d("orange3", x, y, yaw);
			std::cout << "<orange3> x: " << x << ", y: " << y << ", yaw: " << yaw << std::endl;
			simProxy.GetPose2d("orange4", x, y, yaw);
			std::cout << "<orange4> x: " << x << ", y: " << y << ", yaw: " << yaw << std::endl;
		}

		if (laserProxy[90] < laserMaxDetectionRange)
		{
			const int destroyThis = local::findItem(robotNames[0], itemList, itemCount, simProxy);

			// move it out of the simulation
			std::cout << "collecting item" << std::endl;
			simProxy.SetPose2d(itemList[destroyThis].name, -10, -10, 0);
			local::refreshItemList(itemList, simProxy);
		}

		// avoid obstacles
		local::avoidObstacles(forwardSpeed, turnSpeed, sonarProxy);

		// FIXME [delete] >>
		std::cout << "forwardSpeed: " << forwardSpeed << ", turnSpeed: " << turnSpeed << std::endl;

		// set motors
		posProxy.SetSpeed(forwardSpeed, PlayerCc::dtor(turnSpeed));

#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
		Sleep(1000);
#else
		sleep(1);
#endif
	}
}

}  // namespace my_player_stage
