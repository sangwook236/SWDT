//#include "stdafx.h"
#include "PascalVocDataset.h"
#define CV_NO_BACKWARD_COMPATIBILITY
//#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <functional>
#include <sys/stat.h>


//Return the classification ground truth data for all images of a given VOC object class
//--------------------------------------------------------------------------------------
//INPUTS:
// - obj_class          The VOC object class identifier std::string
// - dataset            Specifies whether to extract images from the training or test set
//OUTPUTS:
// - images             An array of ObdImage containing info of all images extracted from the ground truth file
// - object_present     An array of bools specifying whether the object defined by 'obj_class' is present in each image or not
//NOTES:
// This function is primarily useful for the classification task, where only
// whether a given object is present or not in an image is required, and not each object instance's
// position etc.
void PascalVocDataset::getClassImages(const std::string& obj_class, const ObdDatasetType dataset, std::vector<ObdImage>& images, std::vector<char>& object_present)
{
    std::string dataset_str;
    //generate the filename of the classification ground-truth textfile for the object class
    if (dataset == CV_OBD_TRAIN)
    {
        dataset_str = m_train_set;
    } else {
        dataset_str = m_test_set;
    }

    getClassImages_impl(obj_class, dataset_str, images, object_present);
}

void PascalVocDataset::getClassImages_impl(const std::string& obj_class, const std::string& dataset_str, std::vector<ObdImage>& images, std::vector<char>& object_present)
{
    //generate the filename of the classification ground-truth textfile for the object class
    std::string gtFilename = m_class_imageset_path;
    gtFilename.replace(gtFilename.find("%s"),2,obj_class);
    gtFilename.replace(gtFilename.find("%s"),2,dataset_str);

    //parse the ground truth file, storing in two separate vectors
    //for the image code and the ground truth value
    std::vector<std::string> image_codes;
    readClassifierGroundTruth(gtFilename, image_codes, object_present);

    //prepare output arrays
    images.clear();

    convertImageCodesToObdImages(image_codes, images);
}

//Return the object data for all images of a given VOC object class
//-----------------------------------------------------------------
//INPUTS:
// - obj_class          The VOC object class identifier std::string
// - dataset            Specifies whether to extract images from the training or test set
//OUTPUTS:
// - images             An array of ObdImage containing info of all images in chosen dataset (tag, path etc.)
// - objects            Contains the extended object info (bounding box etc.) for each object instance in each image
// - object_data        Contains VOC-specific extended object info (marked difficult etc.)
// - ground_truth       Specifies whether there are any difficult/non-difficult instances of the current
//                          object class within each image
//NOTES:
// This function returns extended object information in addition to the absent/present
// classification data returned by getClassImages. The objects returned for each image in the 'objects'
// array are of all object classes present in the image, and not just the class defined by 'obj_class'.
// 'ground_truth' can be used to determine quickly whether an object instance of the given class is present
// in an image or not.
void PascalVocDataset::getClassObjects(const std::string& obj_class, const ObdDatasetType dataset, std::vector<ObdImage>& images, std::vector<std::vector<ObdObject> >& objects)
{
    std::vector<std::vector<VocObjectData> > object_data;
    std::vector<VocGT> ground_truth;

    getClassObjects(obj_class,dataset,images,objects,object_data,ground_truth);
}

void PascalVocDataset::getClassObjects(const std::string& obj_class, const ObdDatasetType dataset, std::vector<ObdImage>& images, std::vector<std::vector<ObdObject> >& objects, std::vector<std::vector<VocObjectData> >& object_data, std::vector<VocGT>& ground_truth)
{
    //generate the filename of the classification ground-truth textfile for the object class
    std::string gtFilename = m_class_imageset_path;
    gtFilename.replace(gtFilename.find("%s"),2,obj_class);
    if (dataset == CV_OBD_TRAIN)
    {
        gtFilename.replace(gtFilename.find("%s"),2,m_train_set);
    } else {
        gtFilename.replace(gtFilename.find("%s"),2,m_test_set);
    }

    //parse the ground truth file, storing in two separate vectors
    //for the image code and the ground truth value
    std::vector<std::string> image_codes;
    std::vector<char> object_present;
    readClassifierGroundTruth(gtFilename, image_codes, object_present);

    //prepare output arrays
    images.clear();
    objects.clear();
    object_data.clear();
    ground_truth.clear();

    std::string annotationFilename;
    std::vector<ObdObject> image_objects;
    std::vector<VocObjectData> image_object_data;
    VocGT image_gt;

    //transfer to output arrays and read in object data for each image
    for (size_t i = 0; i < image_codes.size(); ++i)
    {
        ObdImage image = getObjects(obj_class, image_codes[i], image_objects, image_object_data, image_gt);

        images.push_back(image);
        objects.push_back(image_objects);
        object_data.push_back(image_object_data);
        ground_truth.push_back(image_gt);
    }
}

//Return ground truth data for the objects present in an image with a given UID
//-----------------------------------------------------------------------------
//INPUTS:
// - id                 VOC Dataset unique identifier (std::string code in form YYYY_XXXXXX where YYYY is the year)
//OUTPUTS:
// - obj_class (*3)     Specifies the object class to use to resolve 'ground_truth'
// - objects            Contains the extended object info (bounding box etc.) for each object in the image
// - object_data (*2,3) Contains VOC-specific extended object info (marked difficult etc.)
// - ground_truth (*3)  Specifies whether there are any difficult/non-difficult instances of the current
//                          object class within the image
//RETURN VALUE:
// ObdImage containing path and other details of image file with given code
//NOTES:
// There are three versions of this function
//  * One returns a simple array of objects given an id [1]
//  * One returns the same as (1) plus VOC specific object data [2]
//  * One returns the same as (2) plus the ground_truth flag. This also requires an extra input obj_class [3]
ObdImage PascalVocDataset::getObjects(const std::string& id, std::vector<ObdObject>& objects)
{
    std::vector<VocObjectData> object_data;
    ObdImage image = getObjects(id, objects, object_data);

    return image;
}

ObdImage PascalVocDataset::getObjects(const std::string& id, std::vector<ObdObject>& objects, std::vector<VocObjectData>& object_data)
{
    //first generate the filename of the annotation file
    std::string annotationFilename = m_annotation_path;

    annotationFilename.replace(annotationFilename.find("%s"),2,id);

    //extract objects contained in the current image from the xml
    extractVocObjects(annotationFilename,objects,object_data);

    //generate image path from extracted std::string code
    std::string path = getImagePath(id);

    ObdImage image(id, path);
    return image;
}

ObdImage PascalVocDataset::getObjects(const std::string& obj_class, const std::string& id, std::vector<ObdObject>& objects, std::vector<VocObjectData>& object_data, VocGT& ground_truth)
{

    //extract object data (except for ground truth flag)
    ObdImage image = getObjects(id,objects,object_data);

    //pregenerate a flag to indicate whether the current class is present or not in the image
    ground_truth = CV_VOC_GT_NONE;
    //iterate through all objects in current image
    for (size_t j = 0; j < objects.size(); ++j)
    {
        if (objects[j].object_class == obj_class)
        {
            if (object_data[j].difficult == false)
            {
                //if at least one non-difficult example is present, this flag is always set to CV_VOC_GT_PRESENT
                ground_truth = CV_VOC_GT_PRESENT;
                break;
            } else {
                //set if at least one object instance is present, but it is marked difficult
                ground_truth = CV_VOC_GT_DIFFICULT;
            }
        }
    }

    return image;
}

//Return ground truth data for the presence/absence of a given object class in an arbitrary array of images
//---------------------------------------------------------------------------------------------------------
//INPUTS:
// - obj_class          The VOC object class identifier std::string
// - images             An array of ObdImage OR strings containing the images for which ground truth
//                          will be computed
//OUTPUTS:
// - ground_truth       An output array indicating the presence/absence of obj_class within each image
void PascalVocDataset::getClassifierGroundTruth(const std::string& obj_class, const std::vector<ObdImage>& images, std::vector<char>& ground_truth)
{
    std::vector<char>(images.size()).swap(ground_truth);

    std::vector<ObdObject> objects;
    std::vector<VocObjectData> object_data;
    std::vector<char>::iterator gt_it = ground_truth.begin();
    for (std::vector<ObdImage>::const_iterator it = images.begin(); it != images.end(); ++it, ++gt_it)
    {
        //getObjects(obj_class, it->id, objects, object_data, voc_ground_truth);
        (*gt_it) = (getClassifierGroundTruthImage(obj_class, it->id));
    }
}

void PascalVocDataset::getClassifierGroundTruth(const std::string& obj_class, const std::vector<std::string>& images, std::vector<char>& ground_truth)
{
    std::vector<char>(images.size()).swap(ground_truth);

    std::vector<ObdObject> objects;
    std::vector<VocObjectData> object_data;
    std::vector<char>::iterator gt_it = ground_truth.begin();
    for (std::vector<std::string>::const_iterator it = images.begin(); it != images.end(); ++it, ++gt_it)
    {
        //getObjects(obj_class, (*it), objects, object_data, voc_ground_truth);
        (*gt_it) = (getClassifierGroundTruthImage(obj_class, (*it)));
    }
}

//Return ground truth data for the accuracy of detection results
//--------------------------------------------------------------
//INPUTS:
// - obj_class          The VOC object class identifier std::string
// - images             An array of ObdImage containing the images for which ground truth
//                          will be computed
// - bounding_boxes     A 2D input array containing the bounding box rects of the objects of
//                          obj_class which were detected in each image
//OUTPUTS:
// - ground_truth       A 2D output array indicating whether each object detection was accurate
//                          or not
// - detection_difficult A 2D output array indicating whether the detection fired on an object
//                          marked as 'difficult'. This allows it to be ignored if necessary
//                          (the voc documentation specifies objects marked as difficult
//                          have no effects on the results and are effectively ignored)
// - (ignore_difficult) If set to true, objects marked as difficult will be ignored when returning
//                          the number of hits for p-r normalization (default = true)
//RETURN VALUE:
//                      Returns the number of object hits in total in the gt to allow proper normalization
//                          of a p-r curve
//NOTES:
// As stated in the VOC documentation, multiple detections of the same object in an image are
// considered FALSE detections e.g. 5 detections of a single object is counted as 1 correct
// detection and 4 false detections - it is the responsibility of the participant's system
// to filter multiple detections from its output
int PascalVocDataset::getDetectorGroundTruth(const std::string& obj_class, const ObdDatasetType dataset, const std::vector<ObdImage>& images, const std::vector<std::vector<cv::Rect> >& bounding_boxes, const std::vector<std::vector<float> >& scores, std::vector<std::vector<char> >& ground_truth, std::vector<std::vector<char> >& detection_difficult, bool ignore_difficult)
{
    int recall_normalization = 0;

    /* first create a list of indices referring to the elements of bounding_boxes and scores in
     * descending order of scores */
    std::vector<ObdScoreIndexSorter> sorted_ids;
    {
        /* first count how many objects to allow preallocation */
        size_t obj_count = 0;
        CV_Assert(images.size() == bounding_boxes.size());
        CV_Assert(scores.size() == bounding_boxes.size());
        for (size_t im_idx = 0; im_idx < scores.size(); ++im_idx)
        {
            CV_Assert(scores[im_idx].size() == bounding_boxes[im_idx].size());
            obj_count += scores[im_idx].size();
        }
        /* preallocate id std::vector */
        sorted_ids.resize(obj_count);
        /* now copy across scores and indexes to preallocated std::vector */
        int flat_pos = 0;
        for (size_t im_idx = 0; im_idx < scores.size(); ++im_idx)
        {
            for (size_t ob_idx = 0; ob_idx < scores[im_idx].size(); ++ob_idx)
            {
                sorted_ids[flat_pos].score = scores[im_idx][ob_idx];
                sorted_ids[flat_pos].image_idx = (int)im_idx;
                sorted_ids[flat_pos].obj_idx = (int)ob_idx;
                ++flat_pos;
            }
        }
        /* and sort the std::vector in descending order of score */
        std::sort(sorted_ids.begin(),sorted_ids.end());
        std::reverse(sorted_ids.begin(),sorted_ids.end());
    }

    /* prepare ground truth + difficult std::vector (1st dimension) */
    std::vector<std::vector<char> >(images.size()).swap(ground_truth);
    std::vector<std::vector<char> >(images.size()).swap(detection_difficult);
    std::vector<std::vector<char> > detected(images.size());

    std::vector<std::vector<ObdObject> > img_objects(images.size());
    std::vector<std::vector<VocObjectData> > img_object_data(images.size());
    /* preload object ground truth bounding box data */
    {
        std::vector<std::vector<ObdObject> > img_objects_all(images.size());
        std::vector<std::vector<VocObjectData> > img_object_data_all(images.size());
        for (size_t image_idx = 0; image_idx < images.size(); ++image_idx)
        {
            /* prepopulate ground truth bounding boxes */
            getObjects(images[image_idx].id, img_objects_all[image_idx], img_object_data_all[image_idx]);
            /* meanwhile, also set length of target ground truth + difficult std::vector to same as number of object detections (2nd dimension) */
            ground_truth[image_idx].resize(bounding_boxes[image_idx].size());
            detection_difficult[image_idx].resize(bounding_boxes[image_idx].size());
        }

        /* save only instances of the object class concerned */
        for (size_t image_idx = 0; image_idx < images.size(); ++image_idx)
        {
            for (size_t obj_idx = 0; obj_idx < img_objects_all[image_idx].size(); ++obj_idx)
            {
                if (img_objects_all[image_idx][obj_idx].object_class == obj_class)
                {
                    img_objects[image_idx].push_back(img_objects_all[image_idx][obj_idx]);
                    img_object_data[image_idx].push_back(img_object_data_all[image_idx][obj_idx]);
                }
            }
            detected[image_idx].resize(img_objects[image_idx].size(), false);
        }
    }

    /* calculate the total number of objects in the ground truth for the current dataset */
    {
        std::vector<ObdImage> gt_images;
        std::vector<char> gt_object_present;
        getClassImages(obj_class, dataset, gt_images, gt_object_present);

        for (size_t image_idx = 0; image_idx < gt_images.size(); ++image_idx)
        {
            std::vector<ObdObject> gt_img_objects;
            std::vector<VocObjectData> gt_img_object_data;
            getObjects(gt_images[image_idx].id, gt_img_objects, gt_img_object_data);
            for (size_t obj_idx = 0; obj_idx < gt_img_objects.size(); ++obj_idx)
            {
                if (gt_img_objects[obj_idx].object_class == obj_class)
                {
                    if ((gt_img_object_data[obj_idx].difficult == false) || (ignore_difficult == false))
                        ++recall_normalization;
                }
            }
        }
    }

#ifdef PR_DEBUG
    int printed_count = 0;
#endif
    /* now iterate through detections in descending order of score, assigning to ground truth bounding boxes if possible */
    for (size_t detect_idx = 0; detect_idx < sorted_ids.size(); ++detect_idx)
    {
        //read in indexes to make following code easier to read
        int im_idx = sorted_ids[detect_idx].image_idx;
        int ob_idx = sorted_ids[detect_idx].obj_idx;
        //set ground truth for the current object to false by default
        ground_truth[im_idx][ob_idx] = false;
        detection_difficult[im_idx][ob_idx] = false;
        float maxov = -1.0;
        bool max_is_difficult = false;
        int max_gt_obj_idx = -1;
        //-- for each detected object iterate through objects present in the bounding box ground truth --
        for (size_t gt_obj_idx = 0; gt_obj_idx < img_objects[im_idx].size(); ++gt_obj_idx)
        {
            if (detected[im_idx][gt_obj_idx] == false)
            {
                //check if the detected object and ground truth object overlap by a sufficient margin
                float ov = testBoundingBoxesForOverlap(bounding_boxes[im_idx][ob_idx], img_objects[im_idx][gt_obj_idx].boundingBox);
                if (ov != -1.0)
                {
                    //if all conditions are met store the overlap score and index (as objects are assigned to the highest scoring match)
                    if (ov > maxov)
                    {
                        maxov = ov;
                        max_gt_obj_idx = (int)gt_obj_idx;
                        //store whether the maximum detection is marked as difficult or not
                        max_is_difficult = (img_object_data[im_idx][gt_obj_idx].difficult);
                    }
                }
            }
        }
        //-- if a match was found, set the ground truth of the current object to true --
        if (maxov != -1.0)
        {
            CV_Assert(max_gt_obj_idx != -1);
            ground_truth[im_idx][ob_idx] = true;
            //store whether the maximum detection was marked as 'difficult' or not
            detection_difficult[im_idx][ob_idx] = max_is_difficult;
            //remove the ground truth object so it doesn't match with subsequent detected objects
            //** this is the behaviour defined by the voc documentation **
            detected[im_idx][max_gt_obj_idx] = true;
        }
#ifdef PR_DEBUG
        if (printed_count < 10)
        {
            cout << printed_count << ": id=" << images[im_idx].id << ", score=" << scores[im_idx][ob_idx] << " (" << ob_idx << ") [" << bounding_boxes[im_idx][ob_idx].x << "," <<
                    bounding_boxes[im_idx][ob_idx].y << "," << bounding_boxes[im_idx][ob_idx].width + bounding_boxes[im_idx][ob_idx].x <<
                    "," << bounding_boxes[im_idx][ob_idx].height + bounding_boxes[im_idx][ob_idx].y << "] detected=" << ground_truth[im_idx][ob_idx] <<
                    ", difficult=" << detection_difficult[im_idx][ob_idx] << endl;
            ++printed_count;
            /* print ground truth */
            for (int gt_obj_idx = 0; gt_obj_idx < img_objects[im_idx].size(); ++gt_obj_idx)
            {
                cout << "    GT: [" << img_objects[im_idx][gt_obj_idx].boundingBox.x << "," <<
                        img_objects[im_idx][gt_obj_idx].boundingBox.y << "," << img_objects[im_idx][gt_obj_idx].boundingBox.width + img_objects[im_idx][gt_obj_idx].boundingBox.x <<
                        "," << img_objects[im_idx][gt_obj_idx].boundingBox.height + img_objects[im_idx][gt_obj_idx].boundingBox.y << "]";
                if (gt_obj_idx == max_gt_obj_idx) cout << " <--- (" << maxov << " overlap)";
                cout << endl;
            }
        }
#endif
    }

    return recall_normalization;
}

//Write VOC-compliant classifier results file
//-------------------------------------------
//INPUTS:
// - obj_class          The VOC object class identifier std::string
// - dataset            Specifies whether working with the training or test set
// - images             An array of ObdImage containing the images for which data will be saved to the result file
// - scores             A corresponding array of confidence scores given a query
// - (competition)      If specified, defines which competition the results are for (see VOC documentation - default 1)
//NOTES:
// The result file path and filename are determined automatically using m_results_directory as a base
void PascalVocDataset::writeClassifierResultsFile( const std::string& out_dir, const std::string& obj_class, const ObdDatasetType dataset, const std::vector<ObdImage>& images, const std::vector<float>& scores, const int competition, const bool overwrite_ifexists)
{
    CV_Assert(images.size() == scores.size());

    std::string output_file_base, output_file;
    if (dataset == CV_OBD_TRAIN)
    {
        output_file_base = out_dir + "/comp" + integerToString(competition) + "_cls_" + m_train_set + "_" + obj_class;
    } else {
        output_file_base = out_dir + "/comp" + integerToString(competition) + "_cls_" + m_test_set + "_" + obj_class;
    }
    output_file = output_file_base + ".txt";

    //check if file exists, and if so create a numbered new file instead
    if (overwrite_ifexists == false)
    {
        struct stat stFileInfo;
        if (stat(output_file.c_str(),&stFileInfo) == 0)
        {
            std::string output_file_new;
            int filenum = 0;
            do
            {
                ++filenum;
                output_file_new = output_file_base + "_" + integerToString(filenum);
                output_file = output_file_new + ".txt";
            } while (stat(output_file.c_str(),&stFileInfo) == 0);
        }
    }

    //output data to file
    std::ofstream result_file(output_file.c_str());
    if (result_file.is_open())
    {
        for (size_t i = 0; i < images.size(); ++i)
        {
            result_file << images[i].id << " " << scores[i] << std::endl;
        }
        result_file.close();
    } else {
        std::string err_msg = "could not open classifier results file '" + output_file + "' for writing. Before running for the first time, a 'results' subdirectory should be created within the VOC dataset base directory. e.g. if the VOC data is stored in /VOC/VOC2010 then the path /VOC/results must be created.";
        CV_Error(cv::Error::StsError,err_msg.c_str());
    }
}

//---------------------------------------
//CALCULATE METRICS FROM VOC RESULTS DATA
//---------------------------------------

//Utility function to construct a VOC-standard classification results filename
//----------------------------------------------------------------------------
//INPUTS:
// - obj_class          The VOC object class identifier std::string
// - task               Specifies whether to generate a filename for the classification or detection task
// - dataset            Specifies whether working with the training or test set
// - (competition)      If specified, defines which competition the results are for (see VOC documentation
//                      default of -1 means this is set to 1 for the classification task and 3 for the detection task)
// - (number)           If specified and above 0, defines which of a number of duplicate results file produced for a given set of
//                      of settings should be used (this number will be added as a postfix to the filename)
//NOTES:
// This is primarily useful for returning the filename of a classification file previously computed using writeClassifierResultsFile
// for example when calling calcClassifierPrecRecall
std::string PascalVocDataset::getResultsFilename(const std::string& obj_class, const VocTask task, const ObdDatasetType dataset, const int competition, const int number)
{
    if ((competition < 1) && (competition != -1))
        CV_Error(cv::Error::StsBadArg,"competition argument should be a positive non-zero number or -1 to accept the default");
    if ((number < 1) && (number != -1))
        CV_Error(cv::Error::StsBadArg,"number argument should be a positive non-zero number or -1 to accept the default");

    std::string dset, task_type;

    if (dataset == CV_OBD_TRAIN)
    {
        dset = m_train_set;
    } else {
        dset = m_test_set;
    }

    int comp = competition;
    if (task == CV_VOC_TASK_CLASSIFICATION)
    {
        task_type = "cls";
        if (comp == -1) comp = 1;
    } else {
        task_type = "det";
        if (comp == -1) comp = 3;
    }

    std::stringstream ss;
    if (number < 1)
    {
        ss << "comp" << comp << "_" << task_type << "_" << dset << "_" << obj_class << ".txt";
    } else {
        ss << "comp" << comp << "_" << task_type << "_" << dset << "_" << obj_class << "_" << number << ".txt";
    }

    std::string filename = ss.str();
    return filename;
}

//Calculate metrics for classification results
//--------------------------------------------
//INPUTS:
// - ground_truth       A std::vector of booleans determining whether the currently tested class is present in each input image
// - scores             A std::vector containing the similarity score for each input image (higher is more similar)
//OUTPUTS:
// - precision          A std::vector containing the precision calculated at each datapoint of a p-r curve generated from the result set
// - recall             A std::vector containing the recall calculated at each datapoint of a p-r curve generated from the result set
// - ap                The ap metric calculated from the result set
// - (ranking)          A std::vector of the same length as 'ground_truth' and 'scores' containing the order of the indices in both of
//                      these arrays when sorting by the ranking score in descending order
//NOTES:
// The result file path and filename are determined automatically using m_results_directory as a base
void PascalVocDataset::calcClassifierPrecRecall(const std::string& obj_class, const std::vector<ObdImage>& images, const std::vector<float>& scores, std::vector<float>& precision, std::vector<float>& recall, float& ap, std::vector<size_t>& ranking)
{
    std::vector<char> res_ground_truth;
    getClassifierGroundTruth(obj_class, images, res_ground_truth);

    calcPrecRecall_impl(res_ground_truth, scores, precision, recall, ap, ranking);
}

void PascalVocDataset::calcClassifierPrecRecall(const std::string& obj_class, const std::vector<ObdImage>& images, const std::vector<float>& scores, std::vector<float>& precision, std::vector<float>& recall, float& ap)
{
    std::vector<char> res_ground_truth;
    getClassifierGroundTruth(obj_class, images, res_ground_truth);

    std::vector<size_t> ranking;
    calcPrecRecall_impl(res_ground_truth, scores, precision, recall, ap, ranking);
}

//< Overloaded version which accepts VOC classification result file input instead of array of scores/ground truth >
//INPUTS:
// - input_file         The path to the VOC standard results file to use for calculating precision/recall
//                      If a full path is not specified, it is assumed this file is in the VOC standard results directory
//                      A VOC standard filename can be retrieved (as used by writeClassifierResultsFile) by calling  getClassifierResultsFilename

void PascalVocDataset::calcClassifierPrecRecall(const std::string& input_file, std::vector<float>& precision, std::vector<float>& recall, float& ap, bool outputRankingFile)
{
    //read in classification results file
    std::vector<std::string> res_image_codes;
    std::vector<float> res_scores;

    std::string input_file_std = checkFilenamePathsep(input_file);
    readClassifierResultsFile(input_file_std, res_image_codes, res_scores);

    //extract the object class and dataset from the results file filename
    std::string class_name, dataset_name;
    extractDataFromResultsFilename(input_file_std, class_name, dataset_name);

    //generate the ground truth for the images extracted from the results file
    std::vector<char> res_ground_truth;

    getClassifierGroundTruth(class_name, res_image_codes, res_ground_truth);

    if (outputRankingFile)
    {
        /* 1. store sorting order by score (descending) in 'order' */
        std::vector<std::pair<size_t, std::vector<float>::const_iterator> > order(res_scores.size());

        size_t n = 0;
        for (std::vector<float>::const_iterator it = res_scores.begin(); it != res_scores.end(); ++it, ++n)
            order[n] = make_pair(n, it);

        std::sort(order.begin(),order.end(),orderingSorter());

        /* 2. save ranking results to text file */
        std::string input_file_std = checkFilenamePathsep(input_file);
        size_t fnamestart = input_file_std.rfind("/");
        std::string scoregt_file_str = input_file_std.substr(0,fnamestart+1) + "scoregt_" + class_name + ".txt";
        std::ofstream scoregt_file(scoregt_file_str.c_str());
        if (scoregt_file.is_open())
        {
            for (size_t i = 0; i < res_scores.size(); ++i)
            {
                scoregt_file << res_image_codes[order[i].first] << " " << res_scores[order[i].first] << " " << res_ground_truth[order[i].first] << std::endl;
            }
            scoregt_file.close();
        } else {
            std::string err_msg = "could not open scoregt file '" + scoregt_file_str + "' for writing.";
            CV_Error(cv::Error::StsError,err_msg.c_str());
        }
    }

    //finally, calculate precision+recall+ap
    std::vector<size_t> ranking;
    calcPrecRecall_impl(res_ground_truth,res_scores,precision,recall,ap,ranking);
}

//< Protected implementation of Precision-Recall calculation used by both calcClassifierPrecRecall and calcDetectorPrecRecall >

void PascalVocDataset::calcPrecRecall_impl(const std::vector<char>& ground_truth, const std::vector<float>& scores, std::vector<float>& precision, std::vector<float>& recall, float& ap, std::vector<size_t>& ranking, int recall_normalization)
{
    CV_Assert(ground_truth.size() == scores.size());

    //add extra element for p-r at 0 recall (in case that first retrieved is positive)
    std::vector<float>(scores.size()+1).swap(precision);
    std::vector<float>(scores.size()+1).swap(recall);

    // SORT RESULTS BY THEIR SCORE
    /* 1. store sorting order in 'order' */
    PascalVocDataset::getSortOrder(scores, ranking);

#ifdef PR_DEBUG
    std::ofstream scoregt_file("D:/pr.txt");
    if (scoregt_file.is_open())
    {
       for (int i = 0; i < scores.size(); ++i)
       {
           scoregt_file << scores[ranking[i]] << " " << ground_truth[ranking[i]] << endl;
       }
       scoregt_file.close();
    }
#endif

    // CALCULATE PRECISION+RECALL

    int retrieved_hits = 0;

    int recall_norm;
    if (recall_normalization != -1)
    {
        recall_norm = recall_normalization;
    } else {
        recall_norm = (int)std::count_if(ground_truth.begin(),ground_truth.end(),std::bind2nd(std::equal_to<bool>(),true));
    }

    ap = 0;
    recall[0] = 0;
    for (size_t idx = 0; idx < ground_truth.size(); ++idx)
    {
        if (ground_truth[ranking[idx]] != 0) ++retrieved_hits;

        precision[idx+1] = static_cast<float>(retrieved_hits)/static_cast<float>(idx+1);
        recall[idx+1] = static_cast<float>(retrieved_hits)/static_cast<float>(recall_norm);

        if (idx == 0)
        {
            //add further point at 0 recall with the same precision value as the first computed point
            precision[idx] = precision[idx+1];
        }
        if (recall[idx+1] == 1.0)
        {
            //if recall = 1, then end early as all positive images have been found
            recall.resize(idx+2);
            precision.resize(idx+2);
            break;
        }
    }

    /* ap calculation */
    if (m_sampled_ap == false)
    {
        // FOR VOC2010+ AP IS CALCULATED FROM ALL DATAPOINTS
        /* make precision monotonically decreasing for purposes of calculating ap */
        std::vector<float> precision_monot(precision.size());
        std::vector<float>::iterator prec_m_it = precision_monot.begin();
        for (std::vector<float>::iterator prec_it = precision.begin(); prec_it != precision.end(); ++prec_it, ++prec_m_it)
        {
            std::vector<float>::iterator max_elem;
            max_elem = std::max_element(prec_it,precision.end());
            (*prec_m_it) = (*max_elem);
        }
        /* calculate ap */
        for (size_t idx = 0; idx < (recall.size()-1); ++idx)
        {
            ap += (recall[idx+1] - recall[idx])*precision_monot[idx+1] +   //no need to take min of prec - is monotonically decreasing
                    0.5f*(recall[idx+1] - recall[idx])*std::abs(precision_monot[idx+1] - precision_monot[idx]);
        }
    } else {
        // FOR BEFORE VOC2010 AP IS CALCULATED BY SAMPLING PRECISION AT RECALL 0.0,0.1,..,1.0

        for (float recall_pos = 0.f; recall_pos <= 1.f; recall_pos += 0.1f)
        {
            //find iterator of the precision corresponding to the first recall >= recall_pos
            std::vector<float>::iterator recall_it = recall.begin();
            std::vector<float>::iterator prec_it = precision.begin();

            while ((*recall_it) < recall_pos)
            {
                ++recall_it;
                ++prec_it;
                if (recall_it == recall.end()) break;
            }

            /* if no recall >= recall_pos found, this level of recall is never reached so stop adding to ap */
            if (recall_it == recall.end()) break;

            /* if the prec_it is valid, compute the max precision at this level of recall or higher */
            std::vector<float>::iterator max_prec = std::max_element(prec_it,precision.end());

            ap += (*max_prec)/11;
        }
    }
}

/* functions for calculating confusion matrix rows */

//Calculate rows of a confusion matrix
//------------------------------------
//INPUTS:
// - obj_class          The VOC object class identifier std::string for the confusion matrix row to compute
// - images             An array of ObdImage containing the images to use for the computation
// - scores             A corresponding array of confidence scores for the presence of obj_class in each image
// - cond               Defines whether to use a cut off point based on recall (CV_VOC_CCOND_RECALL) or score
//                      (CV_VOC_CCOND_SCORETHRESH) the latter is useful for classifier detections where positive
//                      values are positive detections and negative values are negative detections
// - threshold          Threshold value for cond. In case of CV_VOC_CCOND_RECALL, is proportion recall (e.g. 0.5).
//                      In the case of CV_VOC_CCOND_SCORETHRESH is the value above which to count results.
//OUTPUTS:
// - output_headers     An output std::vector of object class headers for the confusion matrix row
// - output_values      An output std::vector of values for the confusion matrix row corresponding to the classes
//                      defined in output_headers
//NOTES:
// The methodology used by the classifier version of this function is that true positives have a single unit
// added to the obj_class column in the confusion matrix row, whereas false positives have a single unit
// distributed in proportion between all the columns in the confusion matrix row corresponding to the objects
// present in the image.
void PascalVocDataset::calcClassifierConfMatRow(const std::string& obj_class, const std::vector<ObdImage>& images, const std::vector<float>& scores, const VocConfCond cond, const float threshold, std::vector<std::string>& output_headers, std::vector<float>& output_values)
{
    CV_Assert(images.size() == scores.size());

    // SORT RESULTS BY THEIR SCORE
    /* 1. store sorting order in 'ranking' */
    std::vector<size_t> ranking;
    PascalVocDataset::getSortOrder(scores, ranking);

    // CALCULATE CONFUSION MATRIX ENTRIES
    /* prepare object category headers */
    output_headers = m_object_classes;
    std::vector<float>(output_headers.size(),0.0).swap(output_values);
    /* find the index of the target object class in the headers for later use */
    int target_idx;
    {
        std::vector<std::string>::iterator target_idx_it = std::find(output_headers.begin(),output_headers.end(),obj_class);
        /* if the target class can not be found, raise an exception */
        if (target_idx_it == output_headers.end())
        {
            std::string err_msg = "could not find the target object class '" + obj_class + "' in list of valid classes.";
            CV_Error(cv::Error::StsError,err_msg.c_str());
        }
        /* convert iterator to index */
        target_idx = std::distance(output_headers.begin(),target_idx_it);
    }

    /* prepare variables related to calculating recall if using the recall threshold */
    int retrieved_hits = 0;
    int total_relevant = 0;
    if (cond == CV_VOC_CCOND_RECALL)
    {
        std::vector<char> ground_truth;
        /* in order to calculate the total number of relevant images for normalization of recall
            it's necessary to extract the ground truth for the images under consideration */
        getClassifierGroundTruth(obj_class, images, ground_truth);
        total_relevant = std::count_if(ground_truth.begin(),ground_truth.end(),std::bind2nd(std::equal_to<bool>(),true));
    }

    /* iterate through images */
    std::vector<ObdObject> img_objects;
    std::vector<VocObjectData> img_object_data;
    int total_images = 0;
    for (size_t image_idx = 0; image_idx < images.size(); ++image_idx)
    {
        /* if using the score as the break condition, check for it now */
        if (cond == CV_VOC_CCOND_SCORETHRESH)
        {
            if (scores[ranking[image_idx]] <= threshold) break;
        }
        /* if continuing for this iteration, increment the image counter for later normalization */
        ++total_images;
        /* for each image retrieve the objects contained */
        getObjects(images[ranking[image_idx]].id, img_objects, img_object_data);
        //check if the tested for object class is present
        if (getClassifierGroundTruthImage(obj_class, images[ranking[image_idx]].id))
        {
            //if the target class is present, assign fully to the target class element in the confusion matrix row
            output_values[target_idx] += 1.0;
            if (cond == CV_VOC_CCOND_RECALL) ++retrieved_hits;
        } else {
            //first delete all objects marked as difficult
            for (size_t obj_idx = 0; obj_idx < img_objects.size(); ++obj_idx)
            {
                if (img_object_data[obj_idx].difficult == true)
                {
                    std::vector<ObdObject>::iterator it1 = img_objects.begin();
                    std::advance(it1,obj_idx);
                    img_objects.erase(it1);
                    std::vector<VocObjectData>::iterator it2 = img_object_data.begin();
                    std::advance(it2,obj_idx);
                    img_object_data.erase(it2);
                    --obj_idx;
                }
            }
            //if the target class is not present, add values to the confusion matrix row in equal proportions to all objects present in the image
            for (size_t obj_idx = 0; obj_idx < img_objects.size(); ++obj_idx)
            {
                //find the index of the currently considered object
                std::vector<std::string>::iterator class_idx_it = std::find(output_headers.begin(),output_headers.end(),img_objects[obj_idx].object_class);
                //if the class name extracted from the ground truth file could not be found in the list of available classes, raise an exception
                if (class_idx_it == output_headers.end())
                {
                    std::string err_msg = "could not find object class '" + img_objects[obj_idx].object_class + "' specified in the ground truth file of '" + images[ranking[image_idx]].id +"'in list of valid classes.";
                    CV_Error(cv::Error::StsError,err_msg.c_str());
                }
                /* convert iterator to index */
                int class_idx = std::distance(output_headers.begin(),class_idx_it);
                //add to confusion matrix row in proportion
                output_values[class_idx] += 1.f/static_cast<float>(img_objects.size());
            }
        }
        //check break conditions if breaking on certain level of recall
        if (cond == CV_VOC_CCOND_RECALL)
        {
            if(static_cast<float>(retrieved_hits)/static_cast<float>(total_relevant) >= threshold) break;
        }
    }
    /* finally, normalize confusion matrix row */
    for (std::vector<float>::iterator it = output_values.begin(); it < output_values.end(); ++it)
    {
        (*it) /= static_cast<float>(total_images);
    }
}

// NOTE: doesn't ignore repeated detections
void PascalVocDataset::calcDetectorConfMatRow(const std::string& obj_class, const ObdDatasetType dataset, const std::vector<ObdImage>& images, const std::vector<std::vector<float> >& scores, const std::vector<std::vector<cv::Rect> >& bounding_boxes, const VocConfCond cond, const float threshold, std::vector<std::string>& output_headers, std::vector<float>& output_values, bool ignore_difficult)
{
    CV_Assert(images.size() == scores.size());
    CV_Assert(images.size() == bounding_boxes.size());

    //collapse scores and ground_truth vectors into 1D vectors to allow ranking
    /* define final flat vectors */
    std::vector<std::string> images_flat;
    std::vector<float> scores_flat;
    std::vector<cv::Rect> bounding_boxes_flat;
    {
        /* first count how many objects to allow preallocation */
        int obj_count = 0;
        CV_Assert(scores.size() == bounding_boxes.size());
        for (size_t img_idx = 0; img_idx < scores.size(); ++img_idx)
        {
            CV_Assert(scores[img_idx].size() == bounding_boxes[img_idx].size());
            for (size_t obj_idx = 0; obj_idx < scores[img_idx].size(); ++obj_idx)
            {
                ++obj_count;
            }
        }
        /* preallocate vectors */
        images_flat.resize(obj_count);
        scores_flat.resize(obj_count);
        bounding_boxes_flat.resize(obj_count);
        /* now copy across to preallocated vectors */
        int flat_pos = 0;
        for (size_t img_idx = 0; img_idx < scores.size(); ++img_idx)
        {
            for (size_t obj_idx = 0; obj_idx < scores[img_idx].size(); ++obj_idx)
            {
                images_flat[flat_pos] = images[img_idx].id;
                scores_flat[flat_pos] = scores[img_idx][obj_idx];
                bounding_boxes_flat[flat_pos] = bounding_boxes[img_idx][obj_idx];
                ++flat_pos;
            }
        }
    }

    // SORT RESULTS BY THEIR SCORE
    /* 1. store sorting order in 'ranking' */
    std::vector<size_t> ranking;
    PascalVocDataset::getSortOrder(scores_flat, ranking);

    // CALCULATE CONFUSION MATRIX ENTRIES
    /* prepare object category headers */
    output_headers = m_object_classes;
    output_headers.push_back("background");
    std::vector<float>(output_headers.size(),0.0).swap(output_values);

    /* prepare variables related to calculating recall if using the recall threshold */
    int retrieved_hits = 0;
    int total_relevant = 0;
    if (cond == CV_VOC_CCOND_RECALL)
    {
//        std::vector<char> ground_truth;
//        /* in order to calculate the total number of relevant images for normalization of recall
//            it's necessary to extract the ground truth for the images under consideration */
//        getClassifierGroundTruth(obj_class, images, ground_truth);
//        total_relevant = std::count_if(ground_truth.begin(),ground_truth.end(),std::bind2nd(std::equal_to<bool>(),true));
        /* calculate the total number of objects in the ground truth for the current dataset */
        std::vector<ObdImage> gt_images;
        std::vector<char> gt_object_present;
        getClassImages(obj_class, dataset, gt_images, gt_object_present);

        for (size_t image_idx = 0; image_idx < gt_images.size(); ++image_idx)
        {
            std::vector<ObdObject> gt_img_objects;
            std::vector<VocObjectData> gt_img_object_data;
            getObjects(gt_images[image_idx].id, gt_img_objects, gt_img_object_data);
            for (size_t obj_idx = 0; obj_idx < gt_img_objects.size(); ++obj_idx)
            {
                if (gt_img_objects[obj_idx].object_class == obj_class)
                {
                    if ((gt_img_object_data[obj_idx].difficult == false) || (ignore_difficult == false))
                        ++total_relevant;
                }
            }
        }
    }

    /* iterate through objects */
    std::vector<ObdObject> img_objects;
    std::vector<VocObjectData> img_object_data;
    int total_objects = 0;
    for (size_t image_idx = 0; image_idx < images.size(); ++image_idx)
    {
        /* if using the score as the break condition, check for it now */
        if (cond == CV_VOC_CCOND_SCORETHRESH)
        {
            if (scores_flat[ranking[image_idx]] <= threshold) break;
        }
        /* increment the image counter for later normalization */
        ++total_objects;
        /* for each image retrieve the objects contained */
        getObjects(images[ranking[image_idx]].id, img_objects, img_object_data);

        //find the ground truth object which has the highest overlap score with the detected object
        float maxov = -1.0;
        int max_gt_obj_idx = -1;
        //-- for each detected object iterate through objects present in ground truth --
        for (size_t gt_obj_idx = 0; gt_obj_idx < img_objects.size(); ++gt_obj_idx)
        {
            //check difficulty flag
            if (ignore_difficult || (img_object_data[gt_obj_idx].difficult == false))
            {
                //if the class matches, then check if the detected object and ground truth object overlap by a sufficient margin
                float ov = testBoundingBoxesForOverlap(bounding_boxes_flat[ranking[image_idx]], img_objects[gt_obj_idx].boundingBox);
                if (ov != -1.f)
                {
                    //if all conditions are met store the overlap score and index (as objects are assigned to the highest scoring match)
                    if (ov > maxov)
                    {
                        maxov = ov;
                        max_gt_obj_idx = gt_obj_idx;
                    }
                }
            }
        }

        //assign to appropriate object class if an object was detected
        if (maxov != -1.0)
        {
            //find the index of the currently considered object
            std::vector<std::string>::iterator class_idx_it = std::find(output_headers.begin(),output_headers.end(),img_objects[max_gt_obj_idx].object_class);
            //if the class name extracted from the ground truth file could not be found in the list of available classes, raise an exception
            if (class_idx_it == output_headers.end())
            {
                std::string err_msg = "could not find object class '" + img_objects[max_gt_obj_idx].object_class + "' specified in the ground truth file of '" + images[ranking[image_idx]].id +"'in list of valid classes.";
                CV_Error(cv::Error::StsError,err_msg.c_str());
            }
            /* convert iterator to index */
            int class_idx = std::distance(output_headers.begin(),class_idx_it);
            //add to confusion matrix row in proportion
            output_values[class_idx] += 1.0;
        } else {
            //otherwise assign to background class
            output_values[output_values.size()-1] += 1.0;
        }

        //check break conditions if breaking on certain level of recall
        if (cond == CV_VOC_CCOND_RECALL)
        {
            if(static_cast<float>(retrieved_hits)/static_cast<float>(total_relevant) >= threshold) break;
        }
    }

    /* finally, normalize confusion matrix row */
    for (std::vector<float>::iterator it = output_values.begin(); it < output_values.end(); ++it)
    {
        (*it) /= static_cast<float>(total_objects);
    }
}

//Save Precision-Recall results to a p-r curve in GNUPlot format
//--------------------------------------------------------------
//INPUTS:
// - output_file        The file to which to save the GNUPlot data file. If only a filename is specified, the data
//                      file is saved to the standard VOC results directory.
// - precision          Vector of precisions as returned from calcClassifier/DetectorPrecRecall
// - recall             Vector of recalls as returned from calcClassifier/DetectorPrecRecall
// - ap                ap as returned from calcClassifier/DetectorPrecRecall
// - (title)            Title to use for the plot (if not specified, just the ap is printed as the title)
//                      This also specifies the filename of the output file if printing to pdf
// - (plot_type)        Specifies whether to instruct GNUPlot to save to a PDF file (CV_VOC_PLOT_PDF) or directly
//                      to screen (CV_VOC_PLOT_SCREEN) in the datafile
//NOTES:
// The GNUPlot data file can be executed using GNUPlot from the commandline in the following way:
//      >> GNUPlot <output_file>
// This will then display the p-r curve on the screen or save it to a pdf file depending on plot_type

void PascalVocDataset::savePrecRecallToGnuplot(const std::string& output_file, const std::vector<float>& precision, const std::vector<float>& recall, const float ap, const std::string title, const VocPlotType plot_type)
{
    std::string output_file_std = checkFilenamePathsep(output_file);

    //if no directory is specified, by default save the output file in the results directory
//    if (output_file_std.find("/") == output_file_std.npos)
//    {
//        output_file_std = m_results_directory + output_file_std;
//    }

    std::ofstream plot_file(output_file_std.c_str());

    if (plot_file.is_open())
    {
        plot_file << "set xrange [0:1]" << std::endl;
        plot_file << "set yrange [0:1]" << std::endl;
        plot_file << "set size square" << std::endl;
        std::string title_text = title;
        if (title_text.size() == 0) title_text = "Precision-Recall Curve";
        plot_file << "set title \"" << title_text << " (ap: " << ap << ")\"" << std::endl;
        plot_file << "set xlabel \"Recall\"" << std::endl;
        plot_file << "set ylabel \"Precision\"" << std::endl;
        plot_file << "set style data lines" << std::endl;
        plot_file << "set nokey" << std::endl;
        if (plot_type == CV_VOC_PLOT_PNG)
        {
            plot_file << "set terminal png" << std::endl;
            std::string pdf_filename;
            if (title.size() != 0)
            {
                pdf_filename = title;
            } else {
                pdf_filename = "prcurve";
            }
            plot_file << "set out \"" << title << ".png\"" << std::endl;
        }
        plot_file << "plot \"-\" using 1:2" << std::endl;
        plot_file << "# X Y" << std::endl;
        CV_Assert(precision.size() == recall.size());
        for (size_t i = 0; i < precision.size(); ++i)
        {
            plot_file << "  " << recall[i] << " " << precision[i] << std::endl;
        }
        plot_file << "end" << std::endl;
        if (plot_type == CV_VOC_PLOT_SCREEN)
        {
            plot_file << "pause -1" << std::endl;
        }
        plot_file.close();
    } else {
        std::string err_msg = "could not open plot file '" + output_file_std + "' for writing.";
        CV_Error(cv::Error::StsError,err_msg.c_str());
    }
}

void PascalVocDataset::readClassifierGroundTruth(const std::string& obj_class, const ObdDatasetType dataset, std::vector<ObdImage>& images, std::vector<char>& object_present)
{
    images.clear();

    std::string gtFilename = m_class_imageset_path;
    gtFilename.replace(gtFilename.find("%s"),2,obj_class);
    if (dataset == CV_OBD_TRAIN)
    {
        gtFilename.replace(gtFilename.find("%s"),2,m_train_set);
    } else {
        gtFilename.replace(gtFilename.find("%s"),2,m_test_set);
    }

    std::vector<std::string> image_codes;
    readClassifierGroundTruth(gtFilename, image_codes, object_present);

    convertImageCodesToObdImages(image_codes, images);
}

void PascalVocDataset::readClassifierResultsFile(const std::string& input_file, std::vector<ObdImage>& images, std::vector<float>& scores)
{
    images.clear();

    std::string input_file_std = checkFilenamePathsep(input_file);

    //if no directory is specified, by default search for the input file in the results directory
//    if (input_file_std.find("/") == input_file_std.npos)
//    {
//        input_file_std = m_results_directory + input_file_std;
//    }

    std::vector<std::string> image_codes;
    readClassifierResultsFile(input_file_std, image_codes, scores);

    convertImageCodesToObdImages(image_codes, images);
}

void PascalVocDataset::readDetectorResultsFile(const std::string& input_file, std::vector<ObdImage>& images, std::vector<std::vector<float> >& scores, std::vector<std::vector<cv::Rect> >& bounding_boxes)
{
    images.clear();

    std::string input_file_std = checkFilenamePathsep(input_file);

    //if no directory is specified, by default search for the input file in the results directory
//    if (input_file_std.find("/") == input_file_std.npos)
//    {
//        input_file_std = m_results_directory + input_file_std;
//    }

    std::vector<std::string> image_codes;
    readDetectorResultsFile(input_file_std, image_codes, scores, bounding_boxes);

    convertImageCodesToObdImages(image_codes, images);
}

const std::vector<std::string>& PascalVocDataset::getObjectClasses()
{
    return m_object_classes;
}

//std::string PascalVocDataset::getResultsDirectory()
//{
//    return m_results_directory;
//}

//---------------------------------------------------------
// Protected Functions ------------------------------------
//---------------------------------------------------------

/*static*/ std::string PascalVocDataset::getVocName( const std::string& vocPath )
{
    size_t found = vocPath.rfind( '/' );
    if( found == std::string::npos )
    {
        found = vocPath.rfind( '\\' );
        if( found == std::string::npos )
            return vocPath;
    }
    return vocPath.substr(found + 1, vocPath.size() - found);
}

void PascalVocDataset::initVoc( const std::string& vocPath, const bool useTestDataset )
{
    initVoc2007to2010( vocPath, useTestDataset );
}

//Initialize file paths and settings for the VOC 2010 dataset
//-----------------------------------------------------------
void PascalVocDataset::initVoc2007to2010( const std::string& vocPath, const bool useTestDataset )
{
    //check format of root directory and modify if necessary

    m_vocName = getVocName( vocPath );

    CV_Assert( !m_vocName.compare("VOC2007") || !m_vocName.compare("VOC2008") ||
               !m_vocName.compare("VOC2009") || !m_vocName.compare("VOC2010") );

    m_vocPath = checkFilenamePathsep( vocPath, true );

    if (useTestDataset)
    {
        m_train_set = "trainval";
        m_test_set = "test";
    } else {
        m_train_set = "train";
        m_test_set = "val";
    }

    // initialize main classification/detection challenge paths
    m_annotation_path = m_vocPath + "/Annotations/%s.xml";
    m_image_path = m_vocPath + "/JPEGImages/%s.jpg";
    m_imageset_path = m_vocPath + "/ImageSets/Main/%s.txt";
    m_class_imageset_path = m_vocPath + "/ImageSets/Main/%s_%s.txt";

    //define available object_classes for VOC2010 dataset
    m_object_classes.push_back("aeroplane");
    m_object_classes.push_back("bicycle");
    m_object_classes.push_back("bird");
    m_object_classes.push_back("boat");
    m_object_classes.push_back("bottle");
    m_object_classes.push_back("bus");
    m_object_classes.push_back("car");
    m_object_classes.push_back("cat");
    m_object_classes.push_back("chair");
    m_object_classes.push_back("cow");
    m_object_classes.push_back("diningtable");
    m_object_classes.push_back("dog");
    m_object_classes.push_back("horse");
    m_object_classes.push_back("motorbike");
    m_object_classes.push_back("person");
    m_object_classes.push_back("pottedplant");
    m_object_classes.push_back("sheep");
    m_object_classes.push_back("sofa");
    m_object_classes.push_back("train");
    m_object_classes.push_back("tvmonitor");

    m_min_overlap = 0.5;

    //up until VOC 2010, ap was calculated by sampling p-r curve, not taking complete curve
    m_sampled_ap = ((m_vocName == "VOC2007") || (m_vocName == "VOC2008") || (m_vocName == "VOC2009"));
}

//Read a VOC classification ground truth text file for a given object class and dataset
//-------------------------------------------------------------------------------------
//INPUTS:
// - filename           The path of the text file to read
//OUTPUTS:
// - image_codes        VOC image codes extracted from the GT file in the form 20XX_XXXXXX where the first four
//                          digits specify the year of the dataset, and the last group specifies a unique ID
// - object_present     For each image in the 'image_codes' array, specifies whether the object class described
//                          in the loaded GT file is present or not
void PascalVocDataset::readClassifierGroundTruth(const std::string& filename, std::vector<std::string>& image_codes, std::vector<char>& object_present)
{
    image_codes.clear();
    object_present.clear();

    std::ifstream gtfile(filename.c_str());
    if (!gtfile.is_open())
    {
        std::string err_msg = "could not open VOC ground truth textfile '" + filename + "'.";
        CV_Error(cv::Error::StsError,err_msg.c_str());
    }

    std::string line;
    std::string image;
    int obj_present;
    while (!gtfile.eof())
    {
        std::getline(gtfile,line);
        std::istringstream iss(line);
        iss >> image >> obj_present;
        if (!iss.fail())
        {
            image_codes.push_back(image);
            object_present.push_back(obj_present == 1);
        } else {
            if (!gtfile.eof()) CV_Error(cv::Error::StsParseError,"error parsing VOC ground truth textfile.");
        }
    }
    gtfile.close();
}

void PascalVocDataset::readClassifierResultsFile(const std::string& input_file, std::vector<std::string>& image_codes, std::vector<float>& scores)
{
    //check if results file exists
    std::ifstream result_file(input_file.c_str());
    if (result_file.is_open())
    {
        std::string line;
        std::string image;
        float score;
        //read in the results file
        while (!result_file.eof())
        {
            std::getline(result_file,line);
            std::istringstream iss(line);
            iss >> image >> score;
            if (!iss.fail())
            {
                image_codes.push_back(image);
                scores.push_back(score);
            } else {
                if(!result_file.eof()) CV_Error(cv::Error::StsParseError,"error parsing VOC classifier results file.");
            }
        }
        result_file.close();
    } else {
        std::string err_msg = "could not open classifier results file '" + input_file + "' for reading.";
        CV_Error(cv::Error::StsError,err_msg.c_str());
    }
}

void PascalVocDataset::readDetectorResultsFile(const std::string& input_file, std::vector<std::string>& image_codes, std::vector<std::vector<float> >& scores, std::vector<std::vector<cv::Rect> >& bounding_boxes)
{
    image_codes.clear();
    scores.clear();
    bounding_boxes.clear();

    //check if results file exists
    std::ifstream result_file(input_file.c_str());
    if (result_file.is_open())
    {
        std::string line;
        std::string image;
        cv::Rect bounding_box;
        float score;
        //read in the results file
        while (!result_file.eof())
        {
            std::getline(result_file,line);
            std::istringstream iss(line);
            iss >> image >> score >> bounding_box.x >> bounding_box.y >> bounding_box.width >> bounding_box.height;
            if (!iss.fail())
            {
                //convert right and bottom positions to width and height
                bounding_box.width -= bounding_box.x;
                bounding_box.height -= bounding_box.y;
                //convert to 0-indexing
                bounding_box.x -= 1;
                bounding_box.y -= 1;
                //store in output vectors
                /* first check if the current image code has been seen before */
                std::vector<std::string>::iterator image_codes_it = std::find(image_codes.begin(),image_codes.end(),image);
                if (image_codes_it == image_codes.end())
                {
                    image_codes.push_back(image);
                    std::vector<float> score_vect(1);
                    score_vect[0] = score;
                    scores.push_back(score_vect);
                    std::vector<cv::Rect> bounding_box_vect(1);
                    bounding_box_vect[0] = bounding_box;
                    bounding_boxes.push_back(bounding_box_vect);
                } else {
                    /* if the image index has been seen before, add the current object below it in the 2D arrays */
                    int image_idx = std::distance(image_codes.begin(),image_codes_it);
                    scores[image_idx].push_back(score);
                    bounding_boxes[image_idx].push_back(bounding_box);
                }
            } else {
                if(!result_file.eof()) CV_Error(cv::Error::StsParseError,"error parsing VOC detector results file.");
            }
        }
        result_file.close();
    } else {
        std::string err_msg = "could not open detector results file '" + input_file + "' for reading.";
        CV_Error(cv::Error::StsError,err_msg.c_str());
    }
}


//Read a VOC annotation xml file for a given image
//------------------------------------------------
//INPUTS:
// - filename           The path of the xml file to read
//OUTPUTS:
// - objects            Array of VocObject describing all object instances present in the given image
void PascalVocDataset::extractVocObjects(const std::string filename, std::vector<ObdObject>& objects, std::vector<VocObjectData>& object_data)
{
#ifdef PR_DEBUG
    int block = 1;
    cout << "SAMPLE VOC OBJECT EXTRACTION for " << filename << ":" << endl;
#endif
    objects.clear();
    object_data.clear();

    std::string contents, object_contents, tag_contents;

    readFileToString(filename, contents);

    //keep on extracting 'object' blocks until no more can be found
    if (extractXMLBlock(contents, "annotation", 0, contents) != -1)
    {
        int searchpos = 0;
        searchpos = extractXMLBlock(contents, "object", searchpos, object_contents);
        while (searchpos != -1)
        {
#ifdef PR_DEBUG
            cout << "SEARCHPOS:" << searchpos << endl;
            cout << "start block " << block << " ---------" << endl;
            cout << object_contents << endl;
            cout << "end block " << block << " -----------" << endl;
            ++block;
#endif

            ObdObject object;
            VocObjectData object_d;

            //object class -------------

            if (extractXMLBlock(object_contents, "name", 0, tag_contents) == -1) CV_Error(cv::Error::StsError,"missing <name> tag in object definition of '" + filename + "'");
            object.object_class.swap(tag_contents);

            //object bounding box -------------

            int xmax, xmin, ymax, ymin;

            if (extractXMLBlock(object_contents, "xmax", 0, tag_contents) == -1) CV_Error(cv::Error::StsError,"missing <xmax> tag in object definition of '" + filename + "'");
            xmax = stringToInteger(tag_contents);

            if (extractXMLBlock(object_contents, "xmin", 0, tag_contents) == -1) CV_Error(cv::Error::StsError,"missing <xmin> tag in object definition of '" + filename + "'");
            xmin = stringToInteger(tag_contents);

            if (extractXMLBlock(object_contents, "ymax", 0, tag_contents) == -1) CV_Error(cv::Error::StsError,"missing <ymax> tag in object definition of '" + filename + "'");
            ymax = stringToInteger(tag_contents);

            if (extractXMLBlock(object_contents, "ymin", 0, tag_contents) == -1) CV_Error(cv::Error::StsError,"missing <ymin> tag in object definition of '" + filename + "'");
            ymin = stringToInteger(tag_contents);

            object.boundingBox.x = xmin-1;      //convert to 0-based indexing
            object.boundingBox.width = xmax - xmin;
            object.boundingBox.y = ymin-1;
            object.boundingBox.height = ymax - ymin;

            CV_Assert(xmin != 0);
            CV_Assert(xmax > xmin);
            CV_Assert(ymin != 0);
            CV_Assert(ymax > ymin);


            //object tags -------------

            if (extractXMLBlock(object_contents, "difficult", 0, tag_contents) != -1)
            {
                object_d.difficult = (tag_contents == "1");
            } else object_d.difficult = false;
            if (extractXMLBlock(object_contents, "occluded", 0, tag_contents) != -1)
            {
                object_d.occluded = (tag_contents == "1");
            } else object_d.occluded = false;
            if (extractXMLBlock(object_contents, "truncated", 0, tag_contents) != -1)
            {
                object_d.truncated = (tag_contents == "1");
            } else object_d.truncated = false;
            if (extractXMLBlock(object_contents, "pose", 0, tag_contents) != -1)
            {
                if (tag_contents == "Frontal") object_d.pose = CV_VOC_POSE_FRONTAL;
                if (tag_contents == "Rear") object_d.pose = CV_VOC_POSE_REAR;
                if (tag_contents == "Left") object_d.pose = CV_VOC_POSE_LEFT;
                if (tag_contents == "Right") object_d.pose = CV_VOC_POSE_RIGHT;
            }

            //add to array of objects
            objects.push_back(object);
            object_data.push_back(object_d);

            //extract next 'object' block from file if it exists
            searchpos = extractXMLBlock(contents, "object", searchpos, object_contents);
        }
    }
}

//Converts an image identifier std::string in the format YYYY_XXXXXX to a single index integer of form XXXXXXYYYY
//where Y represents a year and returns the image path
//----------------------------------------------------------------------------------------------------------
std::string PascalVocDataset::getImagePath(const std::string& input_str)
{
    std::string path = m_image_path;
    path.replace(path.find("%s"),2,input_str);
    return path;
}

//Tests two boundary boxes for overlap (using the intersection over union metric) and returns the overlap if the objects
//defined by the two bounding boxes are considered to be matched according to the criterion outlined in
//the VOC documentation [namely intersection/union > some threshold] otherwise returns -1.0 (no match)
//----------------------------------------------------------------------------------------------------------
float PascalVocDataset::testBoundingBoxesForOverlap(const cv::Rect detection, const cv::Rect ground_truth)
{
    int detection_x2 = detection.x + detection.width;
    int detection_y2 = detection.y + detection.height;
    int ground_truth_x2 = ground_truth.x + ground_truth.width;
    int ground_truth_y2 = ground_truth.y + ground_truth.height;
    //first calculate the boundaries of the intersection of the rectangles
    int intersection_x = std::max(detection.x, ground_truth.x); //rightmost left
    int intersection_y = std::max(detection.y, ground_truth.y); //bottommost top
    int intersection_x2 = std::min(detection_x2, ground_truth_x2); //leftmost right
    int intersection_y2 = std::min(detection_y2, ground_truth_y2); //topmost bottom
    //then calculate the width and height of the intersection rect
    int intersection_width = intersection_x2 - intersection_x + 1;
    int intersection_height = intersection_y2 - intersection_y + 1;
    //if there is no overlap then return false straight away
    if ((intersection_width <= 0) || (intersection_height <= 0)) return -1.0;
    //otherwise calculate the intersection
    int intersection_area = intersection_width*intersection_height;

    //now calculate the union
    int union_area = (detection.width+1)*(detection.height+1) + (ground_truth.width+1)*(ground_truth.height+1) - intersection_area;

    //calculate the intersection over union and use as threshold as per VOC documentation
    float overlap = static_cast<float>(intersection_area)/static_cast<float>(union_area);
    if (overlap > m_min_overlap)
    {
        return overlap;
    } else {
        return -1.0;
    }
}

//Extracts the object class and dataset from the filename of a VOC standard results text file, which takes
//the format 'comp<n>_{cls/det}_<dataset>_<objclass>.txt'
//----------------------------------------------------------------------------------------------------------
void PascalVocDataset::extractDataFromResultsFilename(const std::string& input_file, std::string& class_name, std::string& dataset_name)
{
    std::string input_file_std = checkFilenamePathsep(input_file);

    size_t fnamestart = input_file_std.rfind("/");
    size_t fnameend = input_file_std.rfind(".txt");

    if ((fnamestart == input_file_std.npos) || (fnameend == input_file_std.npos))
        CV_Error(cv::Error::StsError,"Could not extract filename of results file.");

    ++fnamestart;
    if (fnamestart >= fnameend)
        CV_Error(cv::Error::StsError,"Could not extract filename of results file.");

    //extract dataset and class names, triggering exception if the filename format is not correct
    std::string filename = input_file_std.substr(fnamestart, fnameend-fnamestart);
    size_t datasetstart = filename.find("_");
    datasetstart = filename.find("_",datasetstart+1);
    size_t classstart = filename.find("_",datasetstart+1);
    //allow for appended index after a further '_' by discarding this part if it exists
    size_t classend = filename.find("_",classstart+1);
    if (classend == filename.npos) classend = filename.size();
    if ((datasetstart == filename.npos) || (classstart == filename.npos))
        CV_Error(cv::Error::StsError,"Error parsing results filename. Is it in standard format of 'comp<n>_{cls/det}_<dataset>_<objclass>.txt'?");
    ++datasetstart;
    ++classstart;
    if (((datasetstart-classstart) < 1) || ((classend-datasetstart) < 1))
        CV_Error(cv::Error::StsError,"Error parsing results filename. Is it in standard format of 'comp<n>_{cls/det}_<dataset>_<objclass>.txt'?");

    dataset_name = filename.substr(datasetstart,classstart-datasetstart-1);
    class_name = filename.substr(classstart,classend-classstart);
}

bool PascalVocDataset::getClassifierGroundTruthImage(const std::string& obj_class, const std::string& id)
{
    /* if the classifier ground truth data for all images of the current class has not been loaded yet, load it now */
    if (m_classifier_gt_all_ids.empty() || (m_classifier_gt_class != obj_class))
    {
        m_classifier_gt_all_ids.clear();
        m_classifier_gt_all_present.clear();
        m_classifier_gt_class = obj_class;
        for (int i=0; i<2; ++i) //run twice (once over test set and once over training set)
        {
            //generate the filename of the classification ground-truth textfile for the object class
            std::string gtFilename = m_class_imageset_path;
            gtFilename.replace(gtFilename.find("%s"),2,obj_class);
            if (i == 0)
            {
                gtFilename.replace(gtFilename.find("%s"),2,m_train_set);
            } else {
                gtFilename.replace(gtFilename.find("%s"),2,m_test_set);
            }

            //parse the ground truth file, storing in two separate vectors
            //for the image code and the ground truth value
            std::vector<std::string> image_codes;
            std::vector<char> object_present;
            readClassifierGroundTruth(gtFilename, image_codes, object_present);

            m_classifier_gt_all_ids.insert(m_classifier_gt_all_ids.end(),image_codes.begin(),image_codes.end());
            m_classifier_gt_all_present.insert(m_classifier_gt_all_present.end(),object_present.begin(),object_present.end());

            CV_Assert(m_classifier_gt_all_ids.size() == m_classifier_gt_all_present.size());
        }
    }


    //search for the image code
    std::vector<std::string>::iterator it = find (m_classifier_gt_all_ids.begin(), m_classifier_gt_all_ids.end(), id);
    if (it != m_classifier_gt_all_ids.end())
    {
        //image found, so return corresponding ground truth
        return m_classifier_gt_all_present[std::distance(m_classifier_gt_all_ids.begin(),it)] != 0;
    } else {
        std::string err_msg = "could not find classifier ground truth for image '" + id + "' and class '" + obj_class + "'";
        CV_Error(cv::Error::StsError,err_msg.c_str());
    }

    return true;
}

//-------------------------------------------------------------------
// Protected Functions (utility) ------------------------------------
//-------------------------------------------------------------------

//returns a std::vector containing indexes of the input std::vector in sorted ascending/descending order
void PascalVocDataset::getSortOrder(const std::vector<float>& values, std::vector<size_t>& order, bool descending)
{
    /* 1. store sorting order in 'order_pair' */
    std::vector<std::pair<size_t, std::vector<float>::const_iterator> > order_pair(values.size());

    size_t n = 0;
    for (std::vector<float>::const_iterator it = values.begin(); it != values.end(); ++it, ++n)
        order_pair[n] = make_pair(n, it);

    std::sort(order_pair.begin(),order_pair.end(),orderingSorter());
    if (descending == false) std::reverse(order_pair.begin(),order_pair.end());

    std::vector<size_t>(order_pair.size()).swap(order);
    for (size_t i = 0; i < order_pair.size(); ++i)
    {
        order[i] = order_pair[i].first;
    }
}

void PascalVocDataset::readFileToString(const std::string filename, std::string& file_contents)
{
    std::ifstream ifs(filename.c_str());
    if (!ifs) CV_Error(cv::Error::StsError,"could not open text file");

    std::stringstream oss;
    oss << ifs.rdbuf();

    file_contents = oss.str();
}

int PascalVocDataset::stringToInteger(const std::string input_str)
{
    int result;

    std::stringstream ss(input_str);
    if ((ss >> result).fail())
    {
        CV_Error(cv::Error::StsBadArg,"could not perform std::string to integer conversion");
    }
    return result;
}

std::string PascalVocDataset::integerToString(const int input_int)
{
    std::string result;

    std::stringstream ss;
    if ((ss << input_int).fail())
    {
        CV_Error(cv::Error::StsBadArg,"could not perform integer to std::string conversion");
    }
    result = ss.str();
    return result;
}

std::string PascalVocDataset::checkFilenamePathsep( const std::string filename, bool add_trailing_slash )
{
    std::string filename_new = filename;

    size_t pos = filename_new.find("\\\\");
    while (pos != filename_new.npos)
    {
        filename_new.replace(pos,2,"/");
        pos = filename_new.find("\\\\", pos);
    }
    pos = filename_new.find("\\");
    while (pos != filename_new.npos)
    {
        filename_new.replace(pos,2,"/");
        pos = filename_new.find("\\", pos);
    }
    if (add_trailing_slash)
    {
        //add training slash if this is missing
        if (filename_new.rfind("/") != filename_new.length()-1) filename_new += "/";
    }

    return filename_new;
}

void PascalVocDataset::convertImageCodesToObdImages(const std::vector<std::string>& image_codes, std::vector<ObdImage>& images)
{
    images.clear();
    images.reserve(image_codes.size());

    std::string path;
    //transfer to output arrays
    for (size_t i = 0; i < image_codes.size(); ++i)
    {
        //generate image path and indices from extracted std::string code
        path = getImagePath(image_codes[i]);
        images.push_back(ObdImage(image_codes[i], path));
    }
}

//Extract text from within a given tag from an XML file
//-----------------------------------------------------
//INPUTS:
// - src            XML source file
// - tag            XML tag delimiting block to extract
// - searchpos      position within src at which to start search
//OUTPUTS:
// - tag_contents   text extracted between <tag> and </tag> tags
//RETURN VALUE:
// - the position of the final character extracted in tag_contents within src
//      (can be used to call extractXMLBlock recursively to extract multiple blocks)
//      returns -1 if the tag could not be found
int PascalVocDataset::extractXMLBlock(const std::string src, const std::string tag, const int searchpos, std::string& tag_contents)
{
    size_t startpos, next_startpos, endpos;
    int embed_count = 1;

    //find position of opening tag
    startpos = src.find("<" + tag + ">", searchpos);
    if (startpos == std::string::npos) return -1;

    //initialize endpos -
    // start searching for end tag anywhere after opening tag
    endpos = startpos;

    //find position of next opening tag
    next_startpos = src.find("<" + tag + ">", startpos+1);

    //match opening tags with closing tags, and only
    //accept final closing tag of same level as original
    //opening tag
    while (embed_count > 0)
    {
        endpos = src.find("</" + tag + ">", endpos+1);
        if (endpos == std::string::npos) return -1;

        //the next code is only executed if there are embedded tags with the same name
        if (next_startpos != std::string::npos)
        {
            while (next_startpos<endpos)
            {
                //counting embedded start tags
                ++embed_count;
                next_startpos = src.find("<" + tag + ">", next_startpos+1);
                if (next_startpos == std::string::npos) break;
            }
        }
        //passing end tag so decrement nesting level
        --embed_count;
    }

    //finally, extract the tag region
    startpos += tag.length() + 2;
    if (startpos > src.length()) return -1;
    if (endpos > src.length()) return -1;
    tag_contents = src.substr(startpos,endpos-startpos);
    return static_cast<int>(endpos);
}
