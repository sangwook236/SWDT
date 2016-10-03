#include "./include/cv.h"
#include "quickselect.h"

#define MAX_NUM_SAMPLE              8
#define MAX_NUM_DATA                2000
#define max(a,b)                    ((a > b) ? (a) : (b))

int RANSAC(CvPoint2D32f* dataA, CvPoint2D32f* dataB, int dataNumber,
    CvMat* (*EstimateModel)(const CvPoint2D32f*, const CvPoint2D32f*, CvMat*), double (*CalculateError)(CvPoint2D32f*, CvPoint2D32f*, CvMat*),
    CvMat* modelFinal, int numSample, int numIteration, double threshold)
{
    if (dataNumber < numSample) return -1;

    static int sampleIndex[MAX_NUM_SAMPLE];
    static CvPoint2D32f sampleA[MAX_NUM_SAMPLE], sampleB[MAX_NUM_SAMPLE];

    CvMat* model = cvCloneMat(modelFinal);
    double minPenalty = DBL_MAX;
    for (int i = 0; i < numIteration; i++)
    {
        // 1. Sample data
        int sampleCount = 0;
        while (sampleCount < numSample)
        {
            int index = rand() % dataNumber;
            for (int j = 0; j < sampleCount; j++)
                if (sampleIndex[j] == index) break;
            if (j != sampleCount) continue; // Sampling without replacement
            sampleA[sampleCount] = dataA[index];
            sampleB[sampleCount] = dataB[index];
            sampleIndex[sampleCount++] = index;
        }

        // 2. Estimate a model
        EstimateModel(sampleA, sampleB, model);

        // 3. Evaluate the model
        int penalty = 0;
        for (int j = 0; j < dataNumber; j++)
        {
            double error = CalculateError(dataA+j, dataB+j, model);
            if (fabs(error) > threshold) penalty++;
        }
        if (penalty < minPenalty)
        {
            minPenalty = penalty;
            cvCopy(model, modelFinal);
        }
    }

    cvReleaseMat(&model);
    return i;
} // End of 'RANSAC()'

int MLESAC(CvPoint2D32f* dataA, CvPoint2D32f* dataB, int dataNumber,
    CvMat* (*EstimateModel)(const CvPoint2D32f*, const CvPoint2D32f*, CvMat*), double (*CalculateError)(CvPoint2D32f*, CvPoint2D32f*, CvMat*),
    CvMat* finalModel, double* finalGamma,
    int numSample, double errorSpaceSize, int numIteration, double sigma2)
{
    if (dataNumber < numSample) return -1;
    
    static int sampleIndex[MAX_NUM_SAMPLE];
    static CvPoint2D32f sampleA[MAX_NUM_SAMPLE], sampleB[MAX_NUM_SAMPLE];
    static double error2[MAX_NUM_DATA];
    
    CvMat* model = cvCloneMat(finalModel);
    double minSumLogLikelihood = DBL_MAX;
    for (int i = 0; i < numIteration; i++)
    {
        // 1. Sample data
        int sampleCount = 0;
        while (sampleCount < numSample)
        {
            int index = rand() % dataNumber;
            for (int j = 0; j < sampleCount; j++)
                if (sampleIndex[j] == index) break;
                if (j != sampleCount) continue; // Sampling without replacement
                sampleA[sampleCount] = dataA[index];
                sampleB[sampleCount] = dataB[index];
                sampleIndex[sampleCount++] = index;
        }
        
        // 2. Estimate a model and calculate error
        EstimateModel(sampleA, sampleB, model);
        
        // 3 + 4. Common stuff
        for (int j = 0; j < dataNumber; j++)
        {
            double temp = CalculateError(dataA+j, dataB+j, model);
            error2[j] = temp * temp;
        }
        
        // 3. Estimate 'gamma' and 'sigma2' using EM
        double gamma = 0.5;
        for (j = 0; j < 5; j++)
        {
            double sumInlierProb = 0;
            double probOutlier = (1 - gamma) / errorSpaceSize;
            double probInlier_pre = gamma / sqrt(2 * CV_PI * sigma2);
            for (int k = 0; k < dataNumber; k++)
            {
                double probInlier = probInlier_pre * exp(-0.5 * error2[k] / sigma2);
                double probZ = probInlier / (probInlier + probOutlier);
                sumInlierProb += probZ;
            }
            double gamma_pre = gamma;
            gamma = sumInlierProb / dataNumber;
        }
        
        // 4. Evaluate the model
        double sumLogLikelihood = 0;
        double probOutlier = (1 - gamma) / errorSpaceSize;
        double probInlier_pre = gamma / sqrt(2 * CV_PI * sigma2);
        for (j = 0; j < dataNumber; j++)
        {
            double probInlier = probInlier_pre * exp(-0.5 * error2[j] / sigma2);
            double likelihood = probInlier + probOutlier;
            sumLogLikelihood -= log(likelihood); // Negative log likelihood
        }
        if (sumLogLikelihood < minSumLogLikelihood)
        {
            // Update the best model
            minSumLogLikelihood = sumLogLikelihood;
            cvCopy(model, finalModel);
            *finalGamma = gamma;
        }
    }

    return i;
} // End of 'MLESAC()'

