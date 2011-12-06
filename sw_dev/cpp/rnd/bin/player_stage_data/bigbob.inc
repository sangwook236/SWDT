# bigbob's sonars
define bigbobs_sonars ranger
(
	# number of sonars
	scount 4

	# define the pose of each transducer [xpos ypos heading]
	spose[0] [0.75 0.1875 0]  # front left tooth
	spose[1] [0.75 -0.1875 0]  # front right tooth
	spose[2] [0.25 0.5 30]  # left corner
	spose[3] [0.25 -0.5 -30]  # right corner
	
	# define the field of view of each transducer [range_min range_min view_angle]
	sview [0.3 2.0 10]
	
	# define the size of each transducer [xsize ysize] in meters
	ssize [0.01 0.05]
)

# bigbob's blobfinder
define bigbobs_eyes blobfinder
(
	# number of colors to look for
	colors_count 2
	
	# which colors to look for
	colors ["orange" "DarkBlue"]
	
	# camera parameters
	image [160 120]  # resolution
	range 5.00
	fov 1.04719755  # 60 degress = pi/3 radians
)

# bigbob's laser
define bigbobs_laser laser
(
	range_min 0.0
	
	# distance between teeth in meters
	range_max 0.25
	fov 20  # 20 degrees
	
	pose [0.625 0.125 270]
	size [0.025 0.025]
)

# bigbob's body
define bigbob position
(
	# actual size
	size [1.25 1]
	
	# center of rotation offset: offset from its center of area
	origin [0.125 0 0]

	# the shape of bigbob
	polygons 3

	# body
	polygon[0].points 6
	polygon[0].point[0] [0 0]
	polygon[0].point[1] [0 1]
	polygon[0].point[2] [0.75 1]
	polygon[0].point[3] [1 0.75]
	polygon[0].point[4] [1 0.25]
	polygon[0].point[5] [0.75 0]

	# first tooth
	polygon[1].points 4
	polygon[1].point[0] [1 0.75]
	polygon[1].point[1] [1.25 0.75]
	polygon[1].point[2] [1.25 0.625]
	polygon[1].point[3] [1 0.625]

	# second tooth
	polygon[2].points 4
	polygon[2].point[0] [1 0.375]
	polygon[2].point[1] [1.25 0.375]
	polygon[2].point[2] [1.25 0.25]
	polygon[2].point[3] [1 0.25]
	
	# estimated mass in KG
	mass 15.0
	
	# differential steering model
	drive "diff"
	
	# sensors attached to bigbob
	bigbobs_sonars()
	bigbobs_eyes()
	bigbobs_laser()
)
