videoObj = VideoReader('E:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\devel01\M_1.avi');

nFrames = videoObj.NumberOfFrames;
vidHeight = videoObj.Height;
vidWidth = videoObj.Width;

% Preallocate movie structure.
mov(1:nFrames) = struct('cdata', zeros(vidHeight, vidWidth, 3, 'uint8'), 'colormap', []);

% Read one frame at a time.
for k = 1:nFrames
    mov(k).cdata = read(videoObj, k);
end

% Play back the movie once at the video's frame rate.
movie(mov, 1, videoObj.FrameRate);
