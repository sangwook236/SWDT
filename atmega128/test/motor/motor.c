//--TODO: need to add
#define ERROR_SUM_UPPER_LIMIT 1.0f
#define ERROR_SUM_LOWER_LIMIT -1.0f
#define CONTROL_LOOP_OUTPUT_UPPER_LIMIT 1.0f
#define CONTROL_LOOP_OUTPUT_LOWER_LIMIT -1.0f


void readCurrentMotorPose(float *currPos, float *currVel)
{
	//--TODO: need to add
	*currPos = 0.0f;
	*currVel = 0.0f;
}

void writeControlOutput(const float ctrlLoopOutput)
{
}

void readSensors()
{
}

void genTargetMotorPose(float *targetPos, float *targetVel)
{
}

float calcControlOutput(const float targetPos, const float targetVel, const float currPos, const float currVel)
{
	static float errSum = 0.0f;
	const float Kp = 0.0f;
	const float Ki = 0.0f;
	const float Kd = 0.0f;
	const float leakageCoeff = 0.0f;

	const float posErr = targetPos - currPos;
	const float velErr = targetVel - currVel;

	errSum = leakageCoeff * errSum + Ki * posErr;
	if (errSum > ERROR_SUM_UPPER_LIMIT) errSum = ERROR_SUM_UPPER_LIMIT;
	else if (errSum < ERROR_SUM_LOWER_LIMIT) errSum = ERROR_SUM_LOWER_LIMIT;

	const float pid = Kp * posErr + Kd * velErr + errSum;
	const float velFF = 0.0f;
	const float accFF = 0.0f;
	const float jerkFF = 0.0f;

	float ctrlLoopOutput = pid + velFF + accFF + jerkFF;
	if (ctrlLoopOutput > CONTROL_LOOP_OUTPUT_UPPER_LIMIT) ctrlLoopOutput = CONTROL_LOOP_OUTPUT_UPPER_LIMIT;
	else if (ctrlLoopOutput < CONTROL_LOOP_OUTPUT_LOWER_LIMIT) ctrlLoopOutput = CONTROL_LOOP_OUTPUT_LOWER_LIMIT;

	return ctrlLoopOutput;
}

void outputDebugMsg()
{
}

void runControlLoop()
{
	static float ctrlLoopOutput = 0.0f;
	float currPos = 0.0f, currVel = 0.0f;
	float targetPos = 0.0f, targetVel = 0.0f;

	// stage level 1
	readCurrentMotorPose(&currPos, &currVel);
	writeControlOutput(ctrlLoopOutput);
	readSensors();
	// stage level 2
	genTargetMotorPose(&targetPos, &targetVel);
	// stage level 3
	ctrlLoopOutput = calcControlOutput(targetPos, targetVel, currPos, currVel);
	// stage level 4
	outputDebugMsg();
}

