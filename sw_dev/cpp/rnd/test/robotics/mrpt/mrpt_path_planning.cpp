//#include "stdafx.h"
#include <mrpt/slam.h>


namespace {
namespace local {

void path_planning_1()
{
	const std::string GRID_MAP_FILE(".\\robotics_data\\mrpt\\2006-MalagaCampus.gridmap.gz");
	const std::string IMAGE_DEST_FILE(".\\robotics_data\\mrpt\\path_planning.png");

	// load the gridmap:
	if (!mrpt::system::fileExists(GRID_MAP_FILE))
	{
		std::cerr << "map file not found !!!" << std::endl;
		return;
	}

	std::cout << "loading gridmap..." << std::endl;
	mrpt::slam::COccupancyGridMap2D gridmap;
	mrpt::utils::CFileGZInputStream(GRID_MAP_FILE) >> gridmap;
	std::cout << "done! " << (gridmap.getXMax() - gridmap.getXMin()) << " x " << (gridmap.getYMax() - gridmap.getYMin()) << " m" << std::endl;

	// find path:
	mrpt::slam::CPathPlanningCircularRobot pathPlanning;
	pathPlanning.robotRadius = 0.30f;  // [m]

	//mrpt::poses::CPoint2D origin(20, 1);
	const mrpt::poses::CPoint2D origin(20, -110);
	const mrpt::poses::CPoint2D target(90, 40);

	std::cout << "origin: " << origin << std::endl;
	std::cout << "target: " << target << std::endl;

	std::cout << "searching path...";
	std::cout.flush();

	std::deque<mrpt::poses::TPoint2D> thePath;
	mrpt::utils::CTicTac tictac;
	bool notFound;

	tictac.Tic();

	pathPlanning.computePath(gridmap, origin, target, thePath, notFound, 100.0f /* max. distance */);

	double t = tictac.Tac();
	std::cout << "done in " << t*1000 << " ms" << std::endl;

	std::cout << "path found: " << (notFound ? "no" : "yes") << std::endl;
	std::cout << "path has " << (unsigned)thePath.size() << " steps" << std::endl;

	// save result:
	{
		mrpt::utils::CMRPTImage img;
		gridmap.getAsImage(img, false, true);  // force a RGB image

		// draw the path:
		const int radius = mrpt::utils::round(pathPlanning.robotRadius / gridmap.getResolution());

		for (std::deque<mrpt::poses::TPoint2D>::const_iterator it = thePath.begin(); it != thePath.end(); ++it)
			img.drawCircle(gridmap.x2idx(it->x), gridmap.getSizeY() - 1 - gridmap.y2idx(it->y), radius, mrpt::utils::TColor(0, 0, 255));

		img.cross(gridmap.x2idx(origin.x()), gridmap.getSizeY() - 1 - gridmap.y2idx(origin.y()), 0xFF0000, '+', 10);
		img.cross(gridmap.x2idx(target.x()), gridmap.getSizeY() - 1 - gridmap.y2idx(target.y()), 0xFF0000, 'x', 10);

		std::cout << "saving output to: " << IMAGE_DEST_FILE << std::endl;
		img.saveToFile(IMAGE_DEST_FILE);
		std::cout << "done" << std::endl;

#if MRPT_HAS_WXWIDGETS
		{
			mrpt::gui::CDisplayWindow win("computed path");
			win.showImage(img.scaleHalf().scaleHalf());

			win.waitForKey();
		}
#endif
	}
}

int getXIndexInImage(const int &x, const int &width, const bool isRightward)
{
	return isRightward ? x : width - 1 - x;
}

int getYIndexInImage(const int &y, const int &height, const bool isUpward)
{
	return isUpward ? height - 1 - y : y;
}

void path_planning_2()
{
	const bool isRightward = true, isUpward = true;

	//const std::string mapImageFileName(".\\robotics_data\\mrpt\\KAIST-EECS-2ND-INTERSECTION.jpg");
	const std::string mapImageFileName(".\\robotics_data\\mrpt\\KAIST_DemoFile.bmp");

	//
	const float resolution = 0.01f;  // the size of a pixel (cell), [meters]
	const float xCentralPixel = 0.0f;
	const float yCentralPixel = 0.0f;

	//
	mrpt::slam::COccupancyGridMap2D gridMap;
	std::cout << "reading grid map from image file...";
	gridMap.loadFromBitmapFile(mapImageFileName, resolution, xCentralPixel, yCentralPixel);
	//mrpt::utils::CMRPTImage img;
	//img.loadFromFile(mapImageFileName), 
	//gridMap.loadFromBitmap(img, resolution, xCentralPixel, yCentralPixel);
	std::cout << "done!: " << (gridMap.getXMax() - gridMap.getXMin()) << " x " << (gridMap.getYMax() - gridMap.getYMin()) << " m" << std::endl;

	//const int originPt_x = gridMap.getSizeX() / 2, originPt_y = gridMap.getSizeY() / 2;
	const int originPt_x = 0, originPt_y = 0;
	const int targetPt0_x = getXIndexInImage(250, gridMap.getSizeX(), isRightward), targetPt0_y = getYIndexInImage(345, gridMap.getSizeY(), isUpward);
	const int targetPt1_x = getXIndexInImage(1240, gridMap.getSizeX(), isRightward), targetPt1_y = getYIndexInImage(508, gridMap.getSizeY(), isUpward);
	const int targetPt2_x = getXIndexInImage(1423, gridMap.getSizeX(), isRightward), targetPt2_y = getYIndexInImage(451, gridMap.getSizeY(), isUpward);

	// path planning
	std::list<const mrpt::poses::CPoint2D> targets;
	targets.push_back(mrpt::poses::CPoint2D((targetPt0_x - originPt_x) * gridMap.getResolution(), (targetPt0_y - originPt_y) * gridMap.getResolution()));
	targets.push_back(mrpt::poses::CPoint2D((targetPt1_x - originPt_x) * gridMap.getResolution(), (targetPt1_y - originPt_y) * gridMap.getResolution()));
	targets.push_back(mrpt::poses::CPoint2D((targetPt2_x - originPt_x) * gridMap.getResolution(), (targetPt2_y - originPt_y) * gridMap.getResolution()));

	mrpt::slam::CPathPlanningCircularRobot pathPlanning;
	pathPlanning.robotRadius = 0.05f;  // [m]
	std::deque<mrpt::poses::TPoint2D> plannedPath;

	bool isPathGenerated = false;
	if (targets.size() > 2)
	{
		std::list<const mrpt::poses::CPoint2D>::iterator itPrev = targets.begin();
		std::list<const mrpt::poses::CPoint2D>::iterator it = itPrev;
		std::size_t idx = 1;
		for (++it; it != targets.end(); ++it, ++idx)
		{
			bool notFound = true;

			std::deque<mrpt::poses::TPoint2D> aPath;
			pathPlanning.computePath(gridMap, *itPrev, *it, aPath, notFound, /* max. distance */ -1.0f);
			std::cout << idx << "-th path found: " << (notFound ? "no" : "yes") << std::endl;
			if (!notFound)
			{
				isPathGenerated = true;
				std::copy(aPath.begin(), aPath.end(), std::back_inserter(plannedPath));
			}

			itPrev = it;
		}
	}

	//
	mrpt::utils::CMRPTImage imgMap;
	gridMap.getAsImage(imgMap, false, true);  // force a RGB image

	// draw map's coordinate system
	const int x0 = getXIndexInImage(gridMap.x2idx(0.0f), gridMap.getSizeX(), isRightward);
	const int x1 = getXIndexInImage(gridMap.x2idx(1.0f), gridMap.getSizeX(), isRightward);
	const int y0 = getYIndexInImage(gridMap.y2idx(0.0f), gridMap.getSizeY(), isUpward);
	const int y1 = getYIndexInImage(gridMap.y2idx(1.0f), gridMap.getSizeY(), isUpward);
	imgMap.cross(x0, y0, 0xFF0000, 'x', 5, 3);
	imgMap.cross(x1, y0, 0x00FF00, 'x', 5, 3);
	imgMap.cross(x0, y1, 0x0000FF, 'x', 5, 3);
	imgMap.line(x0, y0, x1, y0, 0x00FF00, 5);
	imgMap.line(x0, y0, x0, y1, 0x0000FF, 5);

	// draw target points
	std::size_t idx = 0;
	for (std::list<const mrpt::poses::CPoint2D>::iterator it = targets.begin(); it != targets.end(); ++it, ++idx)
	{
		unsigned int color;
		switch (idx % 3)
		{
		case 0:
			color = 0xFF0000;
			break;
		case 1:
			color = 0x00FF00;
			break;
		case 2:
			color = 0x0000FF;
			break;
		}

		imgMap.cross(getXIndexInImage(gridMap.x2idx(it->x()), gridMap.getSizeX(), isRightward), getYIndexInImage(gridMap.y2idx(it->y()), gridMap.getSizeY(), isUpward), color, '+', 10);
	}

	const int radius = mrpt::utils::round(pathPlanning.robotRadius / gridMap.getResolution());

	// draw the planned path
	for (std::deque<mrpt::poses::TPoint2D>::const_iterator it = plannedPath.begin(); it != plannedPath.end(); ++it)
		imgMap.drawCircle(getXIndexInImage(gridMap.x2idx(it->x), gridMap.getSizeX(), isRightward), getYIndexInImage(gridMap.y2idx(it->y), gridMap.getSizeY(), isUpward), radius, mrpt::utils::TColor(255, 0, 255));

#if MRPT_HAS_WXWIDGETS
	{
		mrpt::gui::CDisplayWindow win("grid map loaded from image file");
		win.showImage(imgMap);
		win.waitForKey();
	}
#endif
}

}  // namespace local
}  // unnamed namespace

namespace my_mrpt {

void path_planning()
{
	//path_planning_1();
	local::path_planning_2();
}

}  // namespace my_mrpt
