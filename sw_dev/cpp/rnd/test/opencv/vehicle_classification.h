1 	#ifndef VEHICLE_CLASSIFICATION_H_GUARD
2 	#define VEHICLE_CLASSIFICATION_H_GUARD
3 	
4 	#include <sstream>
5 	#include <map>
6 	#include <algorithm>
7 	#include <vector>
8 	
9 	#include <cv.h>
10 	#include <cxcore.h>
11 	#include <highgui.h>
12 	#include <ml.h>
13 	
14 	#include "../controller/configuration.h"
15 	
16 	class MVC_Eigencars
17 	{
18 	public:
19 	        MVC_VehicleClassificationParameter* configuration;
20 	        CvMat *car_mean, *car_space;
21 	        CvANN_MLP* neural_network;
22 	        CvSVM* svm;
23 	        IplImage* mask;
24 	        int imsize;
25 	        MVC_Eigencars(MVC_VehicleClassificationParameter* configuration_)
26 	        {
27 	                configuration = configuration_;         
28 	
29 	                //Get mask image
30 	                mask = cvCreateImage(cvSize(configuration->norm_size, configuration->norm_size), IPL_DEPTH_8U, 1);     
31 	                ifstream mask_file("mask.txt");
32 	                for (int i=0; i<configuration->norm_size; ++i)
33 	                        for (int j=0; j<configuration->norm_size; ++j)
34 	                        {
35 	                                int val;
36 	                                mask_file >> val;
37 	                                mask->imageData[i*configuration->norm_size+j] = val;
38 	                        }
39 	               
40 	                //Compute car matrix and car mean vector
41 	                imsize = sqr(configuration->norm_size);
42 	                CvMat* car_matrix = cvCreateMat(configuration->num_car, imsize, CV_32FC1);   
43 	                cvZero(car_matrix);     
44 	                CvMat* car_mean_transpose = cvCreateMat(1, imsize, CV_32FC1);   
45 	                cvZero(car_mean_transpose);     
46 	                IplImage* img = cvCreateImage(cvSize(configuration->norm_size, configuration->norm_size), IPL_DEPTH_8U, 1);     
47 	                for (int c=0; c<configuration->num_car; ++c)
48 	                {
49 	                        stringstream filename;
50 	                        int i = c+1;
51 	                        filename << configuration->car_dir << i << ".bmp";             
52 	                        IplImage* car_image = cvLoadImage(filename.str().c_str(), 0);                   
53 	                        cvResize(car_image, img, CV_INTER_NN);
54 	                        cvReleaseImage(&car_image);                     
55 	                        cvEqualizeHist(img, img);
56 	                        for (int r=0; r<imsize; ++r) {
57 	                                cvmSet(car_matrix, c, r, img->imageData[r]);
58 	                                double val = cvmGet(car_mean_transpose, 0, r) + img->imageData[r];
59 	                                cvmSet(car_mean_transpose, 0, r, val/configuration->num_car);
60 	                        }
61 	                }       
62 	                for (int c=0; c<configuration->num_car; ++c)
63 	                {
64 	                        for (int r=0; r<imsize; ++r)
65 	                        {
66 	                                double val = cvmGet(car_matrix, c, r) - cvmGet(car_mean_transpose, 0, r);
67 	                                cvmSet(car_matrix, c, r, val);                 
68 	                        }
69 	                }
70 	               
71 	                CvMat* evalues = cvCreateMat(1, configuration->num_car, CV_32FC1);     
72 	                CvMat* evectors = cvCreateMat(configuration->num_car, imsize, CV_32FC1);                                       
73 	                cvCalcPCA(car_matrix, car_mean_transpose, evalues, evectors, CV_PCA_USE_AVG);                   
74 	                //Eigenvalues area already sorted
75 	               
76 	                car_space = cvCreateMat(configuration->num_eigen, imsize, CV_32FC1);
77 	                cvZero(car_space);
78 	                for (int c=0; c<configuration->num_eigen; ++c)                 
79 	                        for (int r=0; r<imsize; ++r)
80 	                                cvmSet(car_space, c, r, cvmGet(evectors, c, r));                                       
81 	                car_mean = cvCreateMat(imsize, 1, CV_32FC1);
82 	                cvTranspose(car_mean_transpose, car_mean);
83 	                cvReleaseMat(&car_mean_transpose);
84 	                cvReleaseMat(&evectors);
85 	                cvReleaseMat(&evalues);
86 	                cvReleaseImage(&img);
87 	
88 	                //Training data
89 	                pair<CvMat*, CvMat*> training_data = mvc_GetTrainingData();
90 	
91 	                //Neural network                               
92 	                //mvc_TrainNN(training_data);
93 	                //SVM
94 	                mvc_TrainSVM(training_data);
95 	
96 	                cvReleaseMat(&training_data.first);
97 	                cvReleaseMat(&training_data.second);   
98 	        }
99 	
100 	        pair<CvMat*, CvMat*> mvc_GetTrainingData();
101 	        void mvc_TrainNN(pair<CvMat*, CvMat*> &training_data);
102 	        void mvc_TrainSVM(pair<CvMat*, CvMat*> &training_data);
103 	        CvMat* mvc_ComputeCarWeight(IplImage* &image); 
104 	};
105 	
106 	
107 	CvMat*
108 	MVC_Eigencars::mvc_ComputeCarWeight(IplImage* &image)
109 	{               
110 	        assert(image->imageSize == imsize);
111 	        CvMat *col = cvCreateMat(imsize, 1, CV_32FC1); 
112 	        for (int i=0; i<imsize; ++i)
113 	                cvmSet(col, i, 0, image->imageData[i]);
114 	        cvSub(col, car_mean, col);
115 	        CvMat* mult_res = cvCreateMat(configuration->num_eigen, 1, CV_32FC1);
116 	        cvMatMul(car_space, col, mult_res);
117 	        CvMat* result = cvCreateMat(1, configuration->num_eigen, CV_32FC1);
118 	        cvTranspose(mult_res, result);         
119 	        cvReleaseMat(&mult_res);
120 	        cvReleaseMat(&col);
121 	        return result;
122 	}
123 	
124 	
125 	pair<CvMat*, CvMat*>
126 	MVC_Eigencars::mvc_GetTrainingData()
127 	{
128 	        int ntraining_car = configuration->num_car*2/3;
129 	        int ntraining_noncar = configuration->num_noncar*2/3;
130 	        int ntraining = ntraining_car + ntraining_noncar;
131 	        CvMat* training_inputs = cvCreateMat(ntraining, configuration->num_eigen, CV_32FC1);
132 	        CvMat* training_outputs = cvCreateMat(ntraining, 1, CV_32FC1);
133 	               
134 	        IplImage* img = cvCreateImage(cvSize(configuration->norm_size, configuration->norm_size), IPL_DEPTH_8U, 1);             
135 	        for (int i=1; i<=ntraining_car; ++i)
136 	        {
137 	                //Get the image
138 	                stringstream filename; 
139 	                filename << configuration->car_dir << i << ".bmp";
140 	                IplImage* car_image = cvLoadImage(filename.str().c_str(), 0);   
141 	                cvResize(car_image, img, CV_INTER_NN);
142 	                cvEqualizeHist(img, img);
143 	                cvReleaseImage(&car_image);
144 	                CvMat* weight = mvc_ComputeCarWeight(img);
145 	                for (int j=0; j<configuration->num_eigen; ++j)
146 	                        cvmSet(training_inputs, i-1, j, cvmGet(weight, 0, j));
147 	                cvmSet(training_outputs, i-1, 0, 1);
148 	        }
149 	       
150 	        for (int i=1; i<=ntraining_noncar; ++i)
151 	        {
152 	                //Get the image
153 	                stringstream filename; 
154 	                filename << configuration->noncar_dir << i << ".bmp";
155 	                IplImage* car_image = cvLoadImage(filename.str().c_str(), 0);           
156 	                cvResize(car_image, img, CV_INTER_NN);
157 	                cvReleaseImage(&car_image);                     
158 	                cvEqualizeHist(img, img);
159 	                CvMat* weight = mvc_ComputeCarWeight(img);
160 	                for (int j=0; j<configuration->num_eigen; ++j)
161 	                        cvmSet(training_inputs, ntraining_car + i-1, j, cvmGet(weight, 0, j));
162 	                cvmSet(training_outputs, ntraining_car + i-1, 0, -1);
163 	        }               
164 	        return pair<CvMat*, CvMat*>(training_inputs, training_outputs);
165 	}
166 	
167 	void
168 	MVC_Eigencars::mvc_TrainSVM(pair<CvMat*, CvMat*> &data)
169 	{
170 	        CvSVMParams param;     
171 	        param.term_crit.epsilon = 0.01;
172 	        param.term_crit.max_iter = 50;
173 	        param.term_crit.type = CV_TERMCRIT_ITER|CV_TERMCRIT_EPS;
174 	        param.svm_type = CvSVM::NU_SVC;
175 	        param.kernel_type = CvSVM::RBF;
176 	        param.gamma = 64;
177 	        param.C = 8;
178 	        param.nu = 0.5;
179 	        svm = new CvSVM();
180 	        svm->train(data.first, data.second, 0, 0, param);
181 	}
182 	
183 	void
184 	MVC_Eigencars::mvc_TrainNN(pair<CvMat*, CvMat*> &data)
185 	{
186 	        int a[] = {configuration->num_eigen, 50, 50, 1};
187 	        //3 layers, hidden layer has num_eigen neurons
188 	        CvMat layer_sizes;
189 	        cvInitMatHeader(&layer_sizes, 1, 4, CV_32SC1, a);       
190 	        neural_network = new CvANN_MLP(&layer_sizes, CvANN_MLP::SIGMOID_SYM, 1, 1);
191 	               
192 	        //Create neuron network classification using training examples 
193 	        CvANN_MLP_TrainParams param;   
194 	        param.train_method = CvANN_MLP_TrainParams::BACKPROP;
195 	        param.term_crit.epsilon = 0.01;
196 	        param.term_crit.max_iter = 50;
197 	        param.term_crit.type = CV_TERMCRIT_ITER|CV_TERMCRIT_EPS;
198 	        param.bp_dw_scale = 0.1;
199 	        param.bp_moment_scale = 0.1;           
200 	        neural_network->train(data.first, data.second, NULL, 0, param, 0);
201 	}
202 	
203 	
204 	class MVC_VehicleClassification
205 	{
206 	public:
207 	
208 	        MVC_VehicleClassificationParameter* configuration;
209 	        MVC_Eigencars* eigencars;
210 	        MVC_VehicleClassification(MVC_VehicleClassificationParameter* configuration_)
211 	        {
212 	                configuration = configuration_;
213 	                eigencars = new MVC_Eigencars(configuration);           
214 	        }
215 	       
216 	        void mvc_ComputeEigenvehicle(IplImage* &image, vector<MVC_HLPair> &hl_pairs);
217 	};
218 	
219 	/*
220 	 * Process a more reliable clue to detect cars with the found pairs
221 	 */
222 	void
223 	MVC_VehicleClassification::mvc_ComputeEigenvehicle(IplImage* &image, vector<MVC_HLPair> &hl_pairs)
224 	{
225 	        for (unsigned int i=0; i<hl_pairs.size(); ++i)
226 	        {
227 	                double x = cvRound((hl_pairs[i].hl1.first + hl_pairs[i].hl2.first)/2.0);
228 	                double y = cvRound((hl_pairs[i].hl1.second + hl_pairs[i].hl2.second)/2.0);
229 	                hl_pairs[i].mvc_SetMidPoint(x, y);
230 	                hl_pairs[i].mvc_SetIsCar(false);
231 	
232 	                //Extract a square image from the frame
233 	                int left_bound = (int)(hl_pairs[i].hl1.first - sqr(hl_pairs[i].hlsize));
234 	                int right_bound = (int)(hl_pairs[i].hl1.first + sqr(hl_pairs[i].hlsize));
235 	                int width = right_bound - left_bound;
236 	                int bottom_bound = (int)(max(hl_pairs[i].hl1.second, hl_pairs[i].hl2.second) + 10*hl_pairs[i].hlsize);
237 	                int top_bound = bottom_bound - width;
238 	                IplImage* im = cvCreateImage(cvSize(width, width), IPL_DEPTH_8U, 1);                   
239 	                for (int j=0; j<im->imageSize; ++j)
240 	                        im->imageData[j] = 255;
241 	                for (int c=0; c<width; ++c)
242 	                {
243 	                        for (int r=0; r<width; ++r)
244 	                        {
245 	                                int row = r-1+top_bound;
246 	                                int col = c-1+left_bound;
247 	                                if (mvc_InImage(image, row, col))
248 	                                {                                                                               
249 	                                        int ind = row*image->widthStep + col*image->nChannels;
250 	                                        double val = 0.299*image->imageData[ind]+ 0.5870*image->imageData[ind+1] + 0.114*image->imageData[ind+2];
251 	                                        im->imageData[r*width + c] = cvRound(val);
252 	                                }
253 	                        }
254 	                }
255 	               
256 	                IplImage* im_resize = cvCreateImage(cvSize(configuration->norm_size, configuration->norm_size), IPL_DEPTH_8U, 1);       
257 	                cvResize(im, im_resize, CV_INTER_NN);
258 	                cvReleaseImage(&im);           
259 	                vector<int> values;
260 	                for (int j=0; j<im_resize->imageSize; ++j)
261 	                        if (eigencars->mask->imageData[j] != 0)
262 	                                values.push_back(im_resize->imageData[j]);
263 	                sort(values.begin(), values.end());
264 	                int median = values[(values.size()+1)/2];
265 	
266 	                //Must be a car
267 	                if (median < configuration->EC_black_threshold)                 
268 	                        hl_pairs[i].mvc_SetIsCar(true);
269 	               
270 	                //Appearance checking by eigencars analysis
271 	                else if (median < configuration->EC_bright_threshold)
272 	                {
273 	                        cvEqualizeHist(im_resize, im_resize); //equalization                                           
274 	                        CvMat* car_weights = eigencars->mvc_ComputeCarWeight(im_resize);                       
275 	                        CvMat* result = cvCreateMat(1, 1, CV_32FC1);                   
276 	                        //eigencars->neural_network->predict(car_weights, result);
277 	                        //if (cvmGet(result, 0, 0) >= 0)
278 	                        //      hl_pairs[i].mvc_SetIsCar(true); //car is detected
279 	                        if (eigencars->svm->predict(car_weights) >= 0)
280 	                                hl_pairs[i].mvc_SetIsCar(true); //car is detected
281 	                        cvReleaseMat(&car_weights);
282 	                        cvReleaseMat(&result);
283 	                }
284 	                cvReleaseImage(&im_resize);
285 	        }
286 	}
287 	
288 	#endif
