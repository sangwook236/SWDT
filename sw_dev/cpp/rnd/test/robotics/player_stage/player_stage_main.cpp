//#include "stdafx.h"
#include <libplayerc++/playererror.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_player_stage {

void simple_example();
void rubbish_collecting_robot(int argc, char *argv[]);

}  // namespace my_player_stage

int player_stage_main(int argc, char *argv[])
{
	try
	{
		//my_player_stage::simple_example();
		my_player_stage::rubbish_collecting_robot(argc, argv);
	}
	catch (const PlayerCc::PlayerError &e)
	{
		std::cout << "PlayerCc::PlayerError caught: " << e.GetErrorStr() << std::endl;
		return -1;
	}

	return 0;
}

