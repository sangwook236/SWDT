#if !defined(__OPENCV__PASCAL_VOC_DATASET__H_)
#define __OPENCV__PASCAL_VOC_DATASET__H_ 1

#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/core/core.hpp>
#include <vector>
#include <string>

//-------------------------------------------------------------------
//used to specify the (sub-)dataset over which operations are performed
enum ObdDatasetType {CV_OBD_TRAIN, CV_OBD_TEST};

class ObdObject
{
public:
    std::string object_class;
    cv::Rect boundingBox;
};

//-------------------------------------------------------------------
//extended object data specific to VOC
enum VocPose {CV_VOC_POSE_UNSPECIFIED, CV_VOC_POSE_FRONTAL, CV_VOC_POSE_REAR, CV_VOC_POSE_LEFT, CV_VOC_POSE_RIGHT};

class VocObjectData
{
public:
    bool difficult;
    bool occluded;
    bool truncated;
    VocPose pose;
};

//-------------------------------------------------------------------
//enum VocDataset {CV_VOC2007, CV_VOC2008, CV_VOC2009, CV_VOC2010};
enum VocPlotType {CV_VOC_PLOT_SCREEN, CV_VOC_PLOT_PNG};
enum VocGT {CV_VOC_GT_NONE, CV_VOC_GT_DIFFICULT, CV_VOC_GT_PRESENT};
enum VocConfCond {CV_VOC_CCOND_RECALL, CV_VOC_CCOND_SCORETHRESH};
enum VocTask {CV_VOC_TASK_CLASSIFICATION, CV_VOC_TASK_DETECTION};

//-------------------------------------------------------------------
class ObdImage
{
public:
    ObdImage(std::string p_id, std::string p_path) : id(p_id), path(p_path) {}
    std::string id;
    std::string path;
};

//-------------------------------------------------------------------
//used by getDetectorGroundTruth to sort a two dimensional list of floats in descending order
class ObdScoreIndexSorter
{
public:
    float score;
    int image_idx;
    int obj_idx;
    bool operator < (const ObdScoreIndexSorter& compare) const {return (score < compare.score);}
};

//-------------------------------------------------------------------
class PascalVocDataset
{
public:
    PascalVocDataset( const std::string& vocPath, bool useTestDataset )
        { initVoc( vocPath, useTestDataset ); }
    ~PascalVocDataset(){}
    /* functions for returning classification/object data for multiple images given an object class */
    void getClassImages(const std::string& obj_class, const ObdDatasetType dataset, std::vector<ObdImage>& images, std::vector<char>& object_present);
    void getClassObjects(const std::string& obj_class, const ObdDatasetType dataset, std::vector<ObdImage>& images, std::vector<std::vector<ObdObject> >& objects);
    void getClassObjects(const std::string& obj_class, const ObdDatasetType dataset, std::vector<ObdImage>& images, std::vector<std::vector<ObdObject> >& objects, std::vector<std::vector<VocObjectData> >& object_data, std::vector<VocGT>& ground_truth);
    /* functions for returning object data for a single image given an image id */
    ObdImage getObjects(const std::string& id, std::vector<ObdObject>& objects);
    ObdImage getObjects(const std::string& id, std::vector<ObdObject>& objects, std::vector<VocObjectData>& object_data);
    ObdImage getObjects(const std::string& obj_class, const std::string& id, std::vector<ObdObject>& objects, std::vector<VocObjectData>& object_data, VocGT& ground_truth);
    /* functions for returning the ground truth (present/absent) for groups of images */
    void getClassifierGroundTruth(const std::string& obj_class, const std::vector<ObdImage>& images, std::vector<char>& ground_truth);
    void getClassifierGroundTruth(const std::string& obj_class, const std::vector<std::string>& images, std::vector<char>& ground_truth);
    int getDetectorGroundTruth(const std::string& obj_class, const ObdDatasetType dataset, const std::vector<ObdImage>& images, const std::vector<std::vector<cv::Rect> >& bounding_boxes, const std::vector<std::vector<float> >& scores, std::vector<std::vector<char> >& ground_truth, std::vector<std::vector<char> >& detection_difficult, bool ignore_difficult = true);
    /* functions for writing VOC-compatible results files */
    void writeClassifierResultsFile(const std::string& out_dir, const std::string& obj_class, const ObdDatasetType dataset, const std::vector<ObdImage>& images, const std::vector<float>& scores, const int competition = 1, const bool overwrite_ifexists = false);
    /* functions for calculating metrics from a set of classification/detection results */
    std::string getResultsFilename(const std::string& obj_class, const VocTask task, const ObdDatasetType dataset, const int competition = -1, const int number = -1);
    void calcClassifierPrecRecall(const std::string& obj_class, const std::vector<ObdImage>& images, const std::vector<float>& scores, std::vector<float>& precision, std::vector<float>& recall, float& ap, std::vector<size_t>& ranking);
    void calcClassifierPrecRecall(const std::string& obj_class, const std::vector<ObdImage>& images, const std::vector<float>& scores, std::vector<float>& precision, std::vector<float>& recall, float& ap);
    void calcClassifierPrecRecall(const std::string& input_file, std::vector<float>& precision, std::vector<float>& recall, float& ap, bool outputRankingFile = false);
    /* functions for calculating confusion matrices */
    void calcClassifierConfMatRow(const std::string& obj_class, const std::vector<ObdImage>& images, const std::vector<float>& scores, const VocConfCond cond, const float threshold, std::vector<std::string>& output_headers, std::vector<float>& output_values);
    void calcDetectorConfMatRow(const std::string& obj_class, const ObdDatasetType dataset, const std::vector<ObdImage>& images, const std::vector<std::vector<float> >& scores, const std::vector<std::vector<cv::Rect> >& bounding_boxes, const VocConfCond cond, const float threshold, std::vector<std::string>& output_headers, std::vector<float>& output_values, bool ignore_difficult = true);
    /* functions for outputting gnuplot output files */
    void savePrecRecallToGnuplot(const std::string& output_file, const std::vector<float>& precision, const std::vector<float>& recall, const float ap, const std::string title = std::string(), const VocPlotType plot_type = CV_VOC_PLOT_SCREEN);
    /* functions for reading in result/ground truth files */
    void readClassifierGroundTruth(const std::string& obj_class, const ObdDatasetType dataset, std::vector<ObdImage>& images, std::vector<char>& object_present);
    void readClassifierResultsFile(const std::string& input_file, std::vector<ObdImage>& images, std::vector<float>& scores);
    void readDetectorResultsFile(const std::string& input_file, std::vector<ObdImage>& images, std::vector<std::vector<float> >& scores, std::vector<std::vector<cv::Rect> >& bounding_boxes);
    /* functions for getting dataset info */
    const std::vector<std::string>& getObjectClasses();
    std::string getResultsDirectory();

	static std::string getVocName( const std::string& vocPath );

protected:
    void initVoc( const std::string& vocPath, const bool useTestDataset );
    void initVoc2007to2010( const std::string& vocPath, const bool useTestDataset);
    void readClassifierGroundTruth(const std::string& filename, std::vector<std::string>& image_codes, std::vector<char>& object_present);
    void readClassifierResultsFile(const std::string& input_file, std::vector<std::string>& image_codes, std::vector<float>& scores);
    void readDetectorResultsFile(const std::string& input_file, std::vector<std::string>& image_codes, std::vector<std::vector<float> >& scores, std::vector<std::vector<cv::Rect> >& bounding_boxes);
    void extractVocObjects(const std::string filename, std::vector<ObdObject>& objects, std::vector<VocObjectData>& object_data);
    std::string getImagePath(const std::string& input_str);

    void getClassImages_impl(const std::string& obj_class, const std::string& dataset_str, std::vector<ObdImage>& images, std::vector<char>& object_present);
    void calcPrecRecall_impl(const std::vector<char>& ground_truth, const std::vector<float>& scores, std::vector<float>& precision, std::vector<float>& recall, float& ap, std::vector<size_t>& ranking, int recall_normalization = -1);

    //test two bounding boxes to see if they meet the overlap criteria defined in the VOC documentation
    float testBoundingBoxesForOverlap(const cv::Rect detection, const cv::Rect ground_truth);
    //extract class and dataset name from a VOC-standard classification/detection results filename
    void extractDataFromResultsFilename(const std::string& input_file, std::string& class_name, std::string& dataset_name);
    //get classifier ground truth for a single image
    bool getClassifierGroundTruthImage(const std::string& obj_class, const std::string& id);

    //utility functions
    void getSortOrder(const std::vector<float>& values, std::vector<size_t>& order, bool descending = true);
    int stringToInteger(const std::string input_str);
    void readFileToString(const std::string filename, std::string& file_contents);
    std::string integerToString(const int input_int);
    std::string checkFilenamePathsep(const std::string filename, bool add_trailing_slash = false);
    void convertImageCodesToObdImages(const std::vector<std::string>& image_codes, std::vector<ObdImage>& images);
    int extractXMLBlock(const std::string src, const std::string tag, const int searchpos, std::string& tag_contents);
    //utility sorter
    struct orderingSorter
    {
        bool operator ()(std::pair<size_t, std::vector<float>::const_iterator> const& a, std::pair<size_t, std::vector<float>::const_iterator> const& b)
        {
            return (*a.second) > (*b.second);
        }
    };
    //data members
    std::string m_vocPath;
    std::string m_vocName;
    //std::string m_resPath;

    std::string m_annotation_path;
    std::string m_image_path;
    std::string m_imageset_path;
    std::string m_class_imageset_path;

    std::vector<std::string> m_classifier_gt_all_ids;
    std::vector<char> m_classifier_gt_all_present;
    std::string m_classifier_gt_class;

    //data members
    std::string m_train_set;
    std::string m_test_set;

    std::vector<std::string> m_object_classes;


    float m_min_overlap;
    bool m_sampled_ap;
};

#endif  // __OPENCV__PASCAL_VOC_DATASET__H_
