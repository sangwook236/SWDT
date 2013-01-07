// 下列 ifdef 块是创建使从 DLL 导出更简单的
// 宏的标准方法。此 DLL 中的所有文件都是用命令行上定义的 SOBS_EXPORTS
// 符号编译的。在使用此 DLL 的
// 任何其他项目上不应定义此符号。这样，源文件中包含此文件的任何其他项目都会将
// SOBS_API 函数视为是从 DLL 导入的，而此 DLL 则将用此宏定义的
// 符号视为是被导出的。
//#ifdef SOBS_EXPORTS
//#define SOBS_API __declspec(dllexport)
//#else
//#define SOBS_API __declspec(dllimport)
//#endif


#include "cxcore.h"

typedef struct BestMacth 
{
	int x;
	int y;
	float dist;
	bool isShadow;
}BestMacth;


// 此类是从 sobs.dll 导出的
//class SOBS_API SOBS {
class SOBS {
public:
	static const float COS[360];
	static const float SIN[360];

private:
	int   frameIndex; //Current frame index
	static const int N = 3; //The scale of the background mode
	static const float A[3][3]; //更新矩阵

	CvRect roi;
	//SOBSROI roi;
	int   K;  //学习帧数
	float e1; //训练阶段最佳匹配门限
	float e2; //检测阶段最佳匹配门限
	float c1; //训练阶段背景更新率
	float c2; //检测阶段背景更新率
	///阴影消除相关参数
	float gama;
	float beta;
	float TS;
	float TH;

	int step;

	int iChannel;
	int iHeight;
	int iWidth;
	int iWidthSetp;

	IplImage* scaledImage; //缩放后的图像
	IplImage* foregroundMaskImage; //The object detection result, binary image

	CvMat* backgroundModle;
	CvMat* hsvImage;
	//CvMat* bkMode1; //Self-Organised Neural network background mode
	//CvMat* bkMode2;
	//CvMat* bkMode3;


	bool InitSOBS(IplImage* firstFrame);
	void InitBkMode(CvMat* image);
	BestMacth FindBestMacth(int y, int x, CvMat* image);
	void UpdateBkMode(BestMacth bestMatch, CvMat* image);
	bool restrictROI();
	//bool isShadow(int y, int x, IplImage* image);
public:
	//Constructor function
	SOBS(int pK, int pe1, int pe2, int pc1, int pc2,
		int pgama, int pbeta, int pTS, int pTH);

	bool setROI(CvRect rect);
	CvRect getROI();
	int ProcessFrame(IplImage* image, IplImage* pFrImg=NULL);
	IplImage* GetForegroundMaskImage();
	//IplImage* GetFgmaskImage();
	bool GetFgmaskImage(IplImage* maskImage);
};


