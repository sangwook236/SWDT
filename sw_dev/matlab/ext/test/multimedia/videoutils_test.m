vp = VideoPlayer('E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel01/M_1.avi', 'Verbose', false, 'ShowTime', false);

%% Reproducing the video sequence
% Then we have to create a loop to play the entire video sequence:

while (true)
	plot(vp);

	% Your code here.
	% To access to the current frame use -> vp.Frame
	img = vp.Frame;

	drawnow;
	if (~vp.nextFrame)
       break;
	end
end

%% Releaseing the VideoPlayer Object
% After we have used the *VideoPlayer* object it is necessary to release it
% using this command:

%clear vp;
