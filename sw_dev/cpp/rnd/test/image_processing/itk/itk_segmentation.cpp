#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkGeodesicActiveContourLevelSetImageFilter.h>
#include <itkCurvatureAnisotropicDiffusionImageFilter.h>
#include <itkGradientMagnitudeRecursiveGaussianImageFilter.h>
#include <itkSigmoidImageFilter.h>
#include <itkFastMarchingImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_itk {

// REF [site] >>
//	https://examples.itk.org/src/segmentation/levelsets/segmentwithgeodesicactivecontourlevelset/documentation
//	https://github.com/InsightSoftwareConsortium/ITK/blob/master/Examples/Segmentation/GeodesicActiveContourImageFilter.cxx
void geodesic_active_contour_example()
{
	// REF [site] >> https://github.com/InsightSoftwareConsortium/ITK/tree/master/Examples/Data
	const std::string inputFileName("./BrainProtonDensitySlice.png");
	const std::string outputFileName("./GeodesicActiveContourImageFilterOutput5.png");
	const int seedPosX = 81;
	const int seedPosY = 114;

	const double initialDistance = 5;
	const double sigma = 1.0;
	const double alpha = -0.5;
	const double beta = 3.0;
	const float propagationScaling = 2.0f;
	const size_t numberOfIterations = 1000;
	const auto seedValue = float(-initialDistance);

	constexpr unsigned int Dimension = 2;

	using InputPixelType = float;
	using InputImageType = itk::Image<InputPixelType, Dimension>;
	using OutputPixelType = unsigned char;
	using OutputImageType = itk::Image<OutputPixelType, Dimension>;

	const auto input = itk::ReadImage<InputImageType>(inputFileName);

	using SmoothingFilterType = itk::CurvatureAnisotropicDiffusionImageFilter<InputImageType, InputImageType>;
	auto smoothing = SmoothingFilterType::New();
	smoothing->SetTimeStep(0.125);
	smoothing->SetNumberOfIterations(5);
	smoothing->SetConductanceParameter(9.0);
	smoothing->SetInput(input);

	using GradientFilterType = itk::GradientMagnitudeRecursiveGaussianImageFilter<InputImageType, InputImageType>;
	auto gradientMagnitude = GradientFilterType::New();
	gradientMagnitude->SetSigma(sigma);
	gradientMagnitude->SetInput(smoothing->GetOutput());

	using SigmoidFilterType = itk::SigmoidImageFilter<InputImageType, InputImageType>;
	auto sigmoid = SigmoidFilterType::New();
	sigmoid->SetOutputMinimum(0.0);
	sigmoid->SetOutputMaximum(1.0);
	sigmoid->SetAlpha(alpha);
	sigmoid->SetBeta(beta);
	sigmoid->SetInput(gradientMagnitude->GetOutput());

	using FastMarchingFilterType = itk::FastMarchingImageFilter<InputImageType, InputImageType>;
	auto fastMarching = FastMarchingFilterType::New();

	using GeodesicActiveContourFilterType = itk::GeodesicActiveContourLevelSetImageFilter<InputImageType, InputImageType>;
	auto geodesicActiveContour = GeodesicActiveContourFilterType::New();
	geodesicActiveContour->SetPropagationScaling(propagationScaling);
	geodesicActiveContour->SetCurvatureScaling(1.0);
	geodesicActiveContour->SetAdvectionScaling(1.0);
	geodesicActiveContour->SetMaximumRMSError(0.02);
	geodesicActiveContour->SetNumberOfIterations(numberOfIterations);
	geodesicActiveContour->SetInput(fastMarching->GetOutput());
	geodesicActiveContour->SetFeatureImage(sigmoid->GetOutput());

	using ThresholdingFilterType = itk::BinaryThresholdImageFilter<InputImageType, OutputImageType>;
	auto thresholder = ThresholdingFilterType::New();
	thresholder->SetLowerThreshold(-1000.0);
	thresholder->SetUpperThreshold(0.0);
	thresholder->SetOutsideValue(itk::NumericTraits<OutputPixelType>::min());
	thresholder->SetInsideValue(itk::NumericTraits<OutputPixelType>::max());
	thresholder->SetInput(geodesicActiveContour->GetOutput());

	using NodeContainer = FastMarchingFilterType::NodeContainer;
	using NodeType = FastMarchingFilterType::NodeType;

	InputImageType::IndexType seedPosition;
	seedPosition[0] = seedPosX;
	seedPosition[1] = seedPosY;

	auto seeds = NodeContainer::New();
	NodeType node;
	node.SetValue(seedValue);
	node.SetIndex(seedPosition);

	seeds->Initialize();
	seeds->InsertElement(0, node);

	fastMarching->SetTrialPoints(seeds);
	fastMarching->SetSpeedConstant(1.0);

	using CastFilterType = itk::RescaleIntensityImageFilter<InputImageType, OutputImageType>;

	auto caster1 = CastFilterType::New();
	auto caster2 = CastFilterType::New();
	auto caster3 = CastFilterType::New();
	auto caster4 = CastFilterType::New();

	caster1->SetInput(smoothing->GetOutput());
	caster1->SetOutputMinimum(itk::NumericTraits<OutputPixelType>::min());
	caster1->SetOutputMaximum(itk::NumericTraits<OutputPixelType>::max());
	itk::WriteImage(caster1->GetOutput(), "./GeodesicActiveContourImageFilterOutput1.png");

	caster2->SetInput(gradientMagnitude->GetOutput());
	caster2->SetOutputMinimum(itk::NumericTraits<OutputPixelType>::min());
	caster2->SetOutputMaximum(itk::NumericTraits<OutputPixelType>::max());
	itk::WriteImage(caster2->GetOutput(), "./GeodesicActiveContourImageFilterOutput2.png");

	caster3->SetInput(sigmoid->GetOutput());
	caster3->SetOutputMinimum(itk::NumericTraits<OutputPixelType>::min());
	caster3->SetOutputMaximum(itk::NumericTraits<OutputPixelType>::max());
	itk::WriteImage(caster3->GetOutput(), "./GeodesicActiveContourImageFilterOutput3.png");

	caster4->SetInput(fastMarching->GetOutput());
	caster4->SetOutputMinimum(itk::NumericTraits<OutputPixelType>::min());
	caster4->SetOutputMaximum(itk::NumericTraits<OutputPixelType>::max());

	fastMarching->SetOutputDirection(input->GetDirection());
	fastMarching->SetOutputOrigin(input->GetOrigin());
	fastMarching->SetOutputRegion(input->GetBufferedRegion());
	fastMarching->SetOutputSpacing(input->GetSpacing());

	try
	{
		itk::WriteImage(thresholder->GetOutput(), outputFileName);
	}
	catch (const itk::ExceptionObject &error)
	{
		std::cerr << "Error: " << error << std::endl;
		return;
	}

	std::cout << std::endl;
	std::cout << "Max. no. iterations: " << geodesicActiveContour->GetNumberOfIterations() << std::endl;
	std::cout << "Max. RMS error: " << geodesicActiveContour->GetMaximumRMSError() << std::endl;
	std::cout << std::endl;
	std::cout << "No. elpased iterations: " << geodesicActiveContour->GetElapsedIterations() << std::endl;
	std::cout << "RMS change: " << geodesicActiveContour->GetRMSChange() << std::endl;

	try
	{
		itk::WriteImage(caster4->GetOutput(), "./GeodesicActiveContourImageFilterOutput4.png");
	}
	catch (const itk::ExceptionObject &error)
	{
		std::cerr << "Error: " << error << std::endl;
		return;
	}

	try
	{
		itk::WriteImage(fastMarching->GetOutput(), "./GeodesicActiveContourImageFilterOutput4.mha");
	}
	catch (const itk::ExceptionObject &error)
	{
		std::cerr << "Error: " << error << std::endl;
		return;
	}

	try
	{
		itk::WriteImage(sigmoid->GetOutput(), "./GeodesicActiveContourImageFilterOutput3.mha");
	}
	catch (const itk::ExceptionObject &error)
	{
		std::cerr << "Error: " << error << std::endl;
		return;
	}

	try
	{
		itk::WriteImage(gradientMagnitude->GetOutput(), "./GeodesicActiveContourImageFilterOutput2.mha");
	}
	catch (const itk::ExceptionObject &error)
	{
		std::cerr << "Error: " << error << std::endl;
		return;
	}
}

}  // namespace my_itk
