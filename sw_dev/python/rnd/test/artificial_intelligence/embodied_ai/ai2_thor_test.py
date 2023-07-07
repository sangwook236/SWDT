#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import random, copy
import numpy as np
import ai2thor.controller, ai2thor.util.metrics
from PIL import Image
from matplotlib import pyplot as plt

# REF [site] >> https://colab.research.google.com/drive/1Il6TqmRXOkzYMIEaOU9e4-uTDTIb5Q78
def procthor_example():
	import prior

	# Download ProcTHOR-10k.
	#	#train dataset = 10000, #val dataset = 1000, #test dataset = 1000.
	dataset = prior.load_dataset("procthor-10k")
	print(f"{dataset=}.")
	print(f'{dataset["train"]=}.')

	house = dataset["train"][0]
	print(f"{house.keys()=}")
	print(f"{house=}")

	# Load a house into AI2-THOR.
	house = dataset["train"][3]
	controller = ai2thor.controller.Controller(scene=house)

	# Egocentric images.
	img = Image.fromarray(controller.last_event.frame)
	#img.show()
	plt.imshow(img)
	plt.tight_layout()
	plt.axis("off")
	plt.show()

	# Navigation actions.
	event = controller.step(action="RotateRight")
	img = Image.fromarray(event.frame)
	#img.show()
	plt.imshow(img)
	plt.tight_layout()
	plt.axis("off")
	plt.show()

	event = controller.step(action="RotateRight")
	img = Image.fromarray(event.frame)
	#img.show()
	plt.imshow(img)
	plt.tight_layout()
	plt.axis("off")
	plt.show()

	event = controller.step(action="LookUp")
	img = Image.fromarray(event.frame)
	#img.show()
	plt.imshow(img)
	plt.tight_layout()
	plt.axis("off")
	plt.show()

	event = controller.step(action="MoveAhead")
	img = Image.fromarray(event.frame)
	#img.show()
	plt.imshow(img)
	plt.tight_layout()
	plt.axis("off")
	plt.show()

	# Top-down frame.
	def get_top_down_frame():
		# Setup the top-down camera.
		event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
		pose = copy.deepcopy(event.metadata["actionReturn"])

		bounds = event.metadata["sceneBounds"]["size"]
		max_bound = max(bounds["x"], bounds["z"])

		pose["fieldOfView"] = 50
		pose["position"]["y"] += 1.1 * max_bound
		pose["orthographic"] = False
		pose["farClippingPlane"] = 50
		del pose["orthographicSize"]

		# Add the camera to the scene.
		event = controller.step(
			action="AddThirdPartyCamera",
			**pose,
			skyboxColor="white",
			raise_for_failure=True,
		)
		top_down_frame = event.third_party_camera_frames[-1]
		return Image.fromarray(top_down_frame)

	img = get_top_down_frame()
	#img.show()
	plt.imshow(img)
	plt.tight_layout()
	plt.axis("off")
	plt.show()

	# Randomize the agent's position.
	event = controller.step(action="GetReachablePositions")
	print(f'{event.metadata["actionReturn"]=}.')

	# Visualize these positions (plot them on a scatter plot).
	reachable_positions = event.metadata["actionReturn"]
	xs = [rp["x"] for rp in reachable_positions]
	zs = [rp["z"] for rp in reachable_positions]

	fig, ax = plt.subplots(1, 1)
	ax.scatter(xs, zs)
	ax.set_xlabel("$x$")
	ax.set_ylabel("$z$")
	ax.set_title("Reachable Positions in the Scene")
	ax.set_aspect("equal")
	plt.show()

	# Randomize the agent's position and rotation.
	position = random.choice(reachable_positions)
	rotation = random.choice(range(360))
	print(f"Teleporting the agent to {position} with rotation {rotation}.")

	event = controller.step(action="Teleport", position=position, rotation=rotation)

	img = Image.fromarray(event.frame)
	#img.show()
	plt.imshow(img)
	plt.tight_layout()
	plt.axis("off")
	plt.show()

	# Randomize object materials.
	fig, axs = plt.subplots(2, 5, figsize=(20, 8))

	for ax in axs.flat:
		event = controller.step(action="RandomizeMaterials")
		ax.imshow(event.frame)
		ax.axis("off")
	plt.show()

	# Change houses.
	new_house = dataset["train"][1]
	controller.reset(scene=new_house)

	img = get_top_down_frame()
	#img.show()
	plt.imshow(img)
	plt.tight_layout()
	plt.axis("off")
	plt.show()

# REF [site] >> https://ai2thor.allenai.org/ithor/documentation
def ithor_example():
	# REF [site] >> https://ai2thor.allenai.org/ithor/documentation/initialization

	controller = ai2thor.controller.Controller(
		agentMode="default",
		visibilityDistance=1.5,
		scene="FloorPlan212",

		# Step sizes,
		gridSize=0.25,
		snapToGrid=True,
		rotateStepDegrees=90,

		# Image modalities,
		renderDepthImage=False,
		renderInstanceSegmentation=False,

		# Camera properties,
		width=300,
		height=300,
		fieldOfView=90
	)

	if False:
		controller.reset(scene="FloorPlan319", rotateStepDegrees=30)

		controller.step("PausePhysicsAutoSim")
		controller.step(
			action="AdvancePhysicsStep",
			timeStep=0.01
		)
		controller.step("UnpausePhysicsAutoSim")

	#--------------------
	# REF [site] >> https://ai2thor.allenai.org/ithor/documentation/environment-state

	# Event.
	if False:
		#event = controller.step(...)
		#controller.last_event

		# Metadata.
		#event.metadata
		# ['agent', 'objects', 'arm', 'isSceneAtRest', 'fov', 'heldObjectPose', 'cameraPosition', 'cameraOrthSize', 'thirdPartyCameras', 'collided', 'collidedObjects', 'inventoryObjects', 'sceneName', 'lastAction', 'errorMessage', 'errorCode', 'lastActionSuccess', 'screenWidth', 'screenHeight', 'agentId', 'depthFormat', 'colors', 'flatSurfacesOnGrid', 'distances', 'normals', 'isOpenableGrid', 'segmentedObjectIds', 'objectIdsInBox', 'actionIntReturn', 'actionFloatReturn', 'actionStringsReturn', 'actionFloatsReturn', 'actionVector3sReturn', 'visibleRange', 'currentTime', 'sceneBounds', 'actionReturn'].
		#event.metadata["agent"]  # Agent metadata.
		# ['name', 'cameraHorizon', 'isStanding', 'position', 'rotation', 'inHighFrictionArea'].
		#event.metadata["objects"][i]  # Object metadata.
		# ['name', 'objectType', 'objectId', 'distance', 'visible', 'position', 'rotation', 'axisAlignedBoundingBox', 'objectOrientedBoundingBox', 'mass', 'salientMaterials', 'parentReceptacles', 'receptacle', 'receptacleObjectIds', 'moveable', 'isMoving', 'temperature', 'isInteractable', 'toggleable', 'isToggled', 'breakable', 'isBroken', 'canFillWithLiquid', 'isFilledWithLiquid', 'fillLiquid', 'dirtyable', 'isDirty', 'canBeUsedUp', 'isUsedUp', 'cookable', 'isCooked', 'isHeatSource', 'isColdSource', 'sliceable', 'isSliced', 'openable', 'isOpen', 'openness', 'pickupable', 'isPickedUp', 'assetId', 'controlledObjects'].

		for obj in controller.last_event.metadata["objects"]:
			print(f'{obj["objectType"]}: {obj["objectId"]} - {obj["name"]}: {obj["visible"]}: {obj["position"]} - {obj["rotation"]}.')

	#-----
	# Environment queries.
	if False:
		query = controller.step(
			action="GetObjectInFrame",
			x=0.64,
			y=0.40,
			checkVisible=False
		)
		object_id = query.metadata["actionReturn"]

		query = controller.step(
			action="GetCoordinateFromRaycast",
			x=0.64,
			y=0.40
		)
		coordinate = query.metadata["actionReturn"]

		positions = controller.step(
			action="GetReachablePositions"
		).metadata["actionReturn"]

		event = controller.step(
			action="GetInteractablePoses",
			objectId="Apple|-1.0|+1.0|+1.5",
			positions=[dict(x=0, y=0.9, z=0)],
			rotations=range(0, 360, 10),
			horizons=np.linspace(-30, 60, 30),
			standings=[True, False]
		)
		poses = event.metadata["actionReturn"]

		pose = random.choice(poses)
		controller.step("TeleportFull", **pose)

	#-----
	# Camera.
	if False:
		event = controller.step(
			action="AddThirdPartyCamera",
			position=dict(x=-1.25, y=1, z=-1),
			rotation=dict(x=90, y=0, z=0),
			fieldOfView=90
		)
		event.third_party_camera_frames

		controller.step(
			action="UpdateThirdPartyCamera",
			thirdPartyCameraId=0,
			position=dict(x=-1.25, y=1, z=-1),
			rotation=dict(x=90, y=0, z=0),
			fieldOfView=90
		)

	#--------------------
	# REF [site] >> https://ai2thor.allenai.org/ithor/documentation/scenes

	# Train/Val/Test splits.
	#	It is common to train agents on a subset of scenes, in order to validate and test how well they generalize to new scenes and environments.
	#	For each room type, it is standard practice to treat the first scenes as training scenes, the next scenes as validation scenes, and the last scenes as testing scenes.

	if False:
		# Scene distribution.
		kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
		living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
		bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
		bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

		scenes = kitchens + living_rooms + bedrooms + bathrooms

		# Scene utility.
		controller.ithor_scenes(
			include_kitchens=True,
			include_living_rooms=True,
			include_bedrooms=True,
			include_bathrooms=True
		)

	#--------------------
	# REF [site] >> https://ai2thor.allenai.org/ithor/documentation/objects/object-types

	if False:
		for obj in controller.last_event.metadata["objects"]:
			print(f'{obj["objectType"]}: {obj["objectId"]} - {obj["name"]}: {obj["visible"]}: {obj["position"]} - {obj["rotation"]}.')

	#--------------------
	# REF [site] >> https://ai2thor.allenai.org/ithor/documentation/objects/set-object-states
	if False:
		# Object poses.
		controller.step(
			action='SetObjectPoses',
			objectPoses=[
				{
					"objectName": "Alarm_Clock_19",
					"rotation": {
						"y": 270,
						"x": 0,
						"z": 0
					},
					"position": {
						"y": 0.8197357,
						"x": 2.45610785,
						"z": 0.054755792
					}
				},
				#{...},
				{
					"objectName": "Alarm_Clock_19",
					"rotation": {
						"y": 270,
						"x": 0,
						"z": 0
					},
					"position": {
						"y": 0.8197357,
						"x": 2.64169645,
						"z": -0.134690791
					}
				}
			]
		)

		# Mass properties.
		controller.step(
			action="SetMassProperties",
			objectId="Apple|+1.25|+0.25|-0.75",
			mass=22.5,
			drag=15.9,
			angularDrag=5.6
		)

		# Temperature.
		controller.step(
			action="SetRoomTempDecayTimeForType",
			objectType="Bread",
			TimeUntilRoomTemp=20.0
		)

		controller.step(
			action="SetGlobalRoomTempDecayTime",
			TimeUntilRoomTemp=20.0
		)

		controller.step(
			action='SetDecayTemperatureBool',
			allowDecayTemperature=False
		)

		# Hiding objects.
		controller.step(
			action="RemoveFromScene",
			objectId="Mug|+0.25|-0.27|+1.05"
		)

		controller.step(
			action="DisableObject",
			objectId="DiningTable|+1.0|+1.0|+1.0"
		)

		controller.step(
			action="EnableObject",
			objectId="DiningTable|+1.0|+1.0|+1.0"
		)

	#--------------------
	# REF [site] >> https://ai2thor.allenai.org/ithor/documentation/objects/domain-randomization
	if False:
		# Material randomization.
		controller.step(
			action="RandomizeMaterials",
			useTrainMaterials=None,
			useValMaterials=None,
			useTestMaterials=None,
			inRoomTypes=None
		)

		# Lighting randomization.
		controller.step(
			action="RandomizeLighting",
			brightness=(0.5, 1.5),
			randomizeColor=True,
			hue=(0, 1),
			saturation=(0.5, 1),
			synchronized=False
		)

		# Initial random spawn.
		controller.step(
			action="InitialRandomSpawn",
			randomSeed=0,
			forceVisible=False,
			numPlacementAttempts=5,
			placeStationary=True,
			numDuplicatesOfType = [
				{
					"objectType": "Statue",
					"count": 20
				},
				{
					"objectType": "Bowl",
					"count": 20,
				}
			],
			excludedReceptacles=["CounterTop", "DiningTable"],
			excludedObjectIds=["Apple|1|1|2"]
		)

		# Color randomization.
		controller.step(action="RandomizeColors")

	#--------------------
	# REF [site] >> https://ai2thor.allenai.org/ithor/documentation/navigation
	if False:
		# Movement.
		controller.step(
			action="MoveAhead",
			moveMagnitude=None
		)

		controller.step("MoveBack")
		controller.step("MoveLeft")
		controller.step("MoveRight")

		# Agent rotation.
		controller.step(
			action="RotateRight",
			degrees=None
		)

		controller.step("RotateLeft")

		# Camera rotation.
		controller.step(
			action="LookUp",
			degrees=30
		)

		controller.step("LookDown")

		# Crouch.
		controller.step(action="Crouch")

		# Stand.
		controller.step(action="Stand")

		# Teleport.
		controller.step(
			action="Teleport",
			position=dict(x=1, y=0.9, z=-1.5),
			rotation=dict(x=0, y=270, z=0),
			horizon=30,
			standing=True
		)

		controller.step(
			action="TeleportFull",
			position=dict(x=1, y=0.9, z=-1.5),
			rotation=dict(x=0, y=270, z=0),
			horizon=30,
			standing=True
		)

		# Randomize the position of the agent in the scene, before starting an episode.
		positions = controller.step(
			action="GetReachablePositions"
		).metadata["actionReturn"]

		position = random.choice(positions)
		controller.step(
			action="Teleport",
			position=position
		)

		# Done.
		#	The Done action nothing to the state of the environment. But, it returns a cleaned up event with respect to the metadata.
		#	It is often used in the definition of a successful navigation task (see Anderson et al.), where the agent must call the done action to signal that it knows that it's done, rather than arbitrarily or biasedly guessing.
		#	It is often used to return a cleaned up version of the metadata.
	
		controller.step(action="Done")

	#--------------------
	# REF [site] >> https://ai2thor.allenai.org/ithor/documentation/interactive-physics
	if False:
		# Held objects.

		# Pickup object.
		controller.step(
			action="PickupObject",
			objectId="Apple|1|1|1",
			forceAction=False,
			manualInteract=False
		)

		# Put object.
		controller.step(
			action="PutObject",
			objectId="Apple|1|1|1",
			forceAction=False,
			placeStationary=True
		)

		# Drop object.
		controller.step(
			action="DropHandObject",
			forceAction=False
		)

		# Throw object.
		controller.step(
			action="ThrowObject",
			moveMagnitude=150.0,
			forceAction=False
		)

		# Move held object.
		controller.step(
			action="MoveHeldObjectAhead",
			moveMagnitude=0.1,
			forceVisible=False
		)

		controller.step("MoveHeldObjectBack")
		controller.step("MoveHeldObjectLeft")
		controller.step("MoveHeldObjectRight")
		controller.step("MoveHeldObjectUp")
		controller.step("MoveHeldObjectDown")

		# We also provide a separate helper action, MoveHeldObject, which allows the held object to move in several directions with only a single action:
		controller.step(
			action="MoveHeldObject",
			ahead=0.1,
			right=0.05,
			up=0.12,
			forceVisible=False
		)

		# Rotate held object.

		# Rotate the held object relative to its current rotation.
		controller.step(
			action="RotateHeldObject",
			pitch=90,
			yaw=25,
			roll=45
		)

		# Rotate the held object to a fixed rotation.
		controller.step(
			action="RotateHeldObject",
			rotation=dict(x=90, y=15, z=25)
		)

		#-----
		# Pushing objects.

		# Directional push.
		controller.step(
			action="DirectionalPush",
			objectId="Sofa|3|2|1",
			moveMagnitude="100",
			pushAngle="90"
		)

		# Push object.
		controller.step(
			action="PushObject",
			objectId="Mug|0.25|-0.27",
			forceAction=False
		)

		# Pull object.
		controller.step(
			action="PullObject",
			objectId="Mug|0.25|-0.27",
			forceAction=False
		)

		# Touch then apply force.
		event = controller.step(
			action="TouchThenApplyForce",
			x=0.5,
			y=0.5,
			direction={
				"x": 0.0,
				"y": 1.0,
				"z": 0.0
			},
			moveMagnitude=80,
			handDistance=1.5
		)

		#-----
		# Teleporting objects.

		# Place object at point.
		controller.step(
			action="PlaceObjectAtPoint",
			objectId="Toaster|1|1|1",
			position={
				"x": -1.25,
				"y": 1.0,
				"z": -1.0
			}
		)

		# Get spawn coordinate above receptacle.
		controller.step(
			action="GetSpawnCoordinatesAboveReceptacle",
			objectId="CounterTop|1|1|1",
			anywhere=False
		)

	#--------------------
	# REF [site] >> https://ai2thor.allenai.org/ithor/documentation/object-state-changes
	if False:
		# Open object.
		event = controller.step(
			action="OpenObject",
			objectId="Book|0.25|-0.27|0.95",
			openness=1,
			forceAction=False
		)
		#event.metadata["objects"][i]

		# Close object.
		controller.step(
			action="CloseObject",
			objectId="Book|0.25|-0.27|0.95",
			forceAction=False
		)

		# Break object.
		event = controller.step(
			action="BreakObject",
			objectId="Vase|0.25|0.27|-0.95",
			forceAction=False
		)
		#event.metadata["objects"][i]

		# Cook object.
		event = controller.step(
			action="CookObject",
			objectId="Egg|0.25|-0.27|0.95",
			forceAction=False
		)
		#event.metadata["objects"][i]

		# Slice object.
		event = controller.step(
			action="SliceObject",
			objectId="Potato|0.25|-0.27|0.95",
			forceAction=False
		)
		#event.metadata["objects"][i]

		# Toggle object.
		event = controller.step(
			action="ToggleObjectOn",
			objectId="LightSwitch|0.25|-0.27|0.95",
			forceAction=False
		)
		#event.metadata["objects"][i]

		# Dirty & clean object.
		event = controller.step(
			action="DirtyObject",
			objectId="Mug|0.25|-0.27|0.95",
			forceAction=False
		)
		#event.metadata["objects"][i]

		event = controller.step(
			action="CleanObject",
			objectId="Mug|0.25|-0.27|0.95",
			forceAction=False
		)
		#event.metadata["objects"][i]

		# Fill object with liquid.
		event = controller.step(
			action="FillObjectWithLiquid",
			objectId="Mug|0.25|-0.27|0.95",
			fillLiquid="coffee",
			forceAction=False
		)
		#event.metadata["objects"][i]

		# Empty liquid from object.
		event = controller.step(
			action="EmptyLiquidFromObject",
			objectId="Mug|0.25|-0.27|0.95",
			forceAction=False
		)
		#event.metadata["objects"][i]

		# Use up object.
		event = controller.step(
			action="UseUpObject",
			objectId="ToiletPaper|0.25|-0.27|0.95",
			forceAction=False
		)
		#event.metadata["objects"][i]

# REF [site] >> https://ai2thor.allenai.org/manipulathor/documentation
def manipulathor_example():
	controller = ai2thor.controller.Controller(
		agentMode="arm",
		massThreshold=None,
		scene="FloorPlan203",
		visibilityDistance=1.5,
		gridSize=0.25,
		renderDepthImage=False,
		renderInstanceSegmentation=False,
		width=300,
		height=300,
		fieldOfView=60
	)

	#controller.reset(scene="FloorPlan15", fieldOfView=80)

	#-----
	# Movement.
	if False:
		# Move agent.
		controller.step(
			action="MoveAgent",
			ahead=0.25,
			right=0.25,
			returnToStart=True,
			speed=1,
			fixedDeltaTime=0.02
		)

		# Rotate agent.
		controller.step(
			action="RotateAgent",
			degrees=30,
			returnToStart=True,
			speed=1,
			fixedDeltaTime=0.02
		)

		# Move arm.
		controller.step(
			action="MoveArm",
			position=dict(x=0, y=0.5, z=0),
			coordinateSpace="armBase",
			restrictMovement=False,
			speed=1,
			returnToStart=True,
			fixedDeltaTime=0.02
		)

		# Move arm base.
		controller.step(
			action="MoveArmBase",
			y=0.5,
			speed=1,
			returnToStart=True,
			fixedDeltaTime=0.02
		)

		# Camera rotation.
		controller.step("LookUp")
		controller.step("LookDown")

		# Teleport.
		controller.step(
			action="Teleport",
			position=dict(x=1, y=0.9, z=-1.5),
			rotation=dict(x=0, y=270, z=0),
			horizon=30,
			standing=True
		)

		controller.step(
			action="TeleportFull",
			position=dict(x=1, y=0.9, z=-1.5),
			rotation=dict(x=0, y=270, z=0),
			horizon=30,
			standing=True
		)

		# Randomize the position of the agent in the scene, before starting an episode.
		positions = controller.step(
			action="GetReachablePositions"
		).metadata["actionReturn"]

		position = random.choice(positions)
		controller.step(
			action="Teleport",
			position=position
		)

		# Done.
		#	The Done action nothing to the state of the environment. But, it returns a cleaned up event with respect to the metadata.
		#	It is often used in the definition of a successful navigation task (see Anderson et al.), where the agent must call the done action to signal that it knows that it's done, rather than arbitrarily or biasedly guessing.
		#	It is often used to return a cleaned up version of the metadata.

		controller.step(action="Done")

	#-----
	# Interaction.
	if False:
		# Pickup object.
		controller.step(
			action="PickupObject",
			objectIdCandidates=["Apple|1|1|0", "Fork|2|1|1"]
		)

		# Release object.
		controller.step(action="ReleaseObject")

		# Set hand sphere radius.
		controller.step(
			action="SetHandSphereRadius",
			radius=0.1
		)

	#-----
	# Environment state.

	# Events.
	#event = controller.step(...)
	#controller.last_event

	# Metadata.
	#event.metadata
	#event.metadata["arm"]  # Arm metadata.
	#event.metadata["arm"]["joints"][i]  # Joint metadata.
	#event.metadata["agent"]  # Agent metadata.
	#event.metadata["objects"]  # Object metadata.

# REF [site] >> https://ai2thor.allenai.org/robothor/documentation
def robothor_example():
	controller = ai2thor.controller.Controller(
		agentMode="locobot",
		visibilityDistance=1.5,
		scene="FloorPlan_Train1_3",
		gridSize=0.25,
		movementGaussianSigma=0.005,
		rotateStepDegrees=90,
		rotateGaussianSigma=0.5,
		renderDepthImage=False,
		renderInstanceSegmentation=False,
		width=300,
		height=300,
		fieldOfView=60
	)

	#controller.reset(scene="FloorPlan_Train7_5", rotateStepDegrees=30)

	#-----
	# Navigation.
	if False:
		# Movement.
		controller.step(
			action="MoveAhead",
			moveMagnitude=0.25
		)

		controller.step("MoveBack")

		# Agent rotation.
		controller.step(
			action="RotateRight",
			degrees=90
		)

		controller.step("RotateLeft")

		# Camera rotation.
		controller.step("LookUp")
		controller.step("LookDown")

		# Teleport.
		controller.step(
			action="Teleport",
			position=dict(x=0.999, y=1.01, z=-0.3541),
			rotation=dict(x=0, y=90, z=0),
			horizon=30
		)

		controller.step(
			action="TeleportFull",
			position=dict(x=0.999, y=1.01, z=-0.3541),
			rotation=dict(x=0, y=90, z=0),
			horizon=30
		)

		# Get reachable positions.
		positions = controller.step(
			action="GetReachablePositions"
		).metadata["actionReturn"]

		position = random.choice(positions)
		controller.step(
			action="Teleport",
			position=position
		)

		# Done.
		#	The Done action nothing to the state of the environment. But, it returns a cleaned up event with respect to the metadata.
		#	It is often used in the definition of a successful navigation task (see Anderson et al.), where the agent must call the done action to signal that it knows that it's done, rather than arbitrarily or biasedly guessing.
		#	It is often used to return a cleaned up version of the metadata.
	
		controller.step(action="Done")

	#-----
	# Environment state.

	# Events.
	#event = controller.step(...)
	#controller.last_event

	# Metadata.
	#event.metadata
	#event.metadata["agent"]  # Agent metadata.
	#event.metadata["objects"]  # Object metadata.

	#-----
	# Evaluation.
	if False:
		# Object types.
		TARGET_OBJECT_TYPES = {
			"AlarmClock,"
			"Apple,"
			"BaseballBat,"
			"BasketBall,"
			"Bowl,"
			"GarbageCan,"
			"HousePlant,"
			"Laptop,"
			"Mug,"
			"RemoteControl,"
			"SprayBottle,"
			"Television,"
			"Vase"
		}

		BACKGROUND_OBJECT_TYPES = {
			"ArmChair",
			"Bed",
			"Book",
			"Bottle",
			"Box",
			"ButterKnife",
			"Candle",
			"CD",
			"CellPhone",
			"Chair",
			"CoffeeTable",
			"Cup",
			"DeskLamp",
			"Desk",
			"DiningTable",
			"Drawer",
			"Dresser",
			"FloorLamp",
			"Fork",
			"Newspaper",
			"Painting",
			"Pencil",
			"Pen",
			"PepperShaker",
			"Pillow",
			"Plate",
			"Pot",
			"SaltShaker",
			"Shelf",
			"SideTable",
			"Sofa",
			"Statue",
			"TeddyBear",
			"TennisRacket",
			"TVStand",
			"Watch"
		}

		# Shortest path.
		path = ai2thor.util.metrics.get_shortest_path_to_object_type(
			controller=controller,
			object_type="Apple",
			initial_position=dict(
				x=0,
				y=0.9,
				z=0.25
			)
		)

		# Path distance.
		ai2thor.util.metrics.path_distance(path)

		# Success weighted by (normalized inverse) Path Length (SPL).
		ai2thor.util.metrics.compute_single_spl(
			path,
			shortest_path,
			successful_path
		)

def simulation_test():
	#import torch
	import prior

	if True:
		# #train dataset = 10000, #val dataset = 1000, #test dataset = 1000.
		dataset = prior.load_dataset("procthor-10k")

		# Load a house into AI2-THOR.
		house = dataset["train"][3]
		controller = ai2thor.controller.Controller(
			scene=house,
			width=800,
			height=800,
			fieldOfView=45
		)
	elif False:
		# #train dataset = 0, #val dataset = 5300, #test dataset = 3511.
		dataset = prior.load_dataset("object-nav-eval")

		house = dataset["val"][3]
		controller = ai2thor.controller.Controller(scene=house)
	else:
		controller = ai2thor.controller.Controller(
			agentMode="default",
			visibilityDistance=1.5,
			scene="FloorPlan212",

			# Step sizes,
			gridSize=0.25,
			snapToGrid=True,
			rotateStepDegrees=90,

			# Image modalities,
			renderDepthImage=False,
			renderInstanceSegmentation=False,

			# Camera properties,
			width=300,
			height=300,
			fieldOfView=90
		)

	if False:
		model_path = prior.load_model(project="procthor-models", model="object-nav-pretraining")
		torch.load(model_path)

	#for obj in controller.last_event.metadata["objects"]:
	#	print(f'{obj["objectType"]}: {obj["objectId"]} - {obj["name"]}: {obj["visible"]}: {obj["position"]} - {obj["rotation"]}.')
	print(f'#objects = {len(controller.last_event.metadata["objects"])}.')

	# Randomize the position of the agent in the scene, before starting an episode.
	reachable_positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
	#reachable_positions = controller.last_event.metadata["actionReturn"]  # None.
	assert reachable_positions

	if False:
		# Visualize these positions (plot them on a scatter plot).
		obj_xs = [obj["position"]["x"] for obj in controller.last_event.metadata["objects"]]
		obj_zs = [obj["position"]["z"] for obj in controller.last_event.metadata["objects"]]
		xs = [rp["x"] for rp in reachable_positions]
		zs = [rp["z"] for rp in reachable_positions]

		plt.scatter(xs, zs)
		plt.scatter(obj_xs, obj_zs, color="red")
		plt.xlabel("$x$")
		plt.ylabel("$z$")
		plt.title("Reachable Positions in the Scene")
		plt.tight_layout()
		plt.show()

	position = random.choice(reachable_positions)
	controller.step(
		action="Teleport",
		position=position
	)

	#while True:
	#	pass

def main():
	# Install.
	#	pip install ai2thor prior

	#procthor_example()  # ProcTHOR.

	#ithor_example()  # iTHOR.
	#manipulathor_example()  # ManipulaTHOR.
	#robothor_example()  # RoboTHOR.

	#-----
	simulation_test()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
