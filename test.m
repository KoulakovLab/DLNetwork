%TEST performs testing of your network
%
%   Sergey Shuvaev, 2016. sshuvaev@cshl.edu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Your images and labels here %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Loading
images = loadMNISTImages(fullfile('mnist','t10k-images-idx3-ubyte'));
labels = loadMNISTLabels(fullfile('mnist','t10k-labels-idx1-ubyte'));

images_tmp = reshape(images, [28, 28, size(images, 2)]); %Preprocessing
images = zeros(32, 32, size(images, 2));
images(3 : 30, 3 : 30, :) = images_tmp; %Zero padding
images = images * 1.275 - 0.1; %Mean 0, variance 1
clear images_tmp;

%%%%%%%%%%%
% Testing %
%%%%%%%%%%%

pos = 0; %number of positively classified images
len = length(leNet);
tic;

for i = 1 : size(images, 3) %Detecting
    
    fprintf('%d\n', i);
    
    leNet(1).setInput(images(:, :, i)); %Initializing
    
    for j = 2 : len - 1
        stepForward(leNet, j); %Forward prop
    end
    
    if leNet(len - 1).getLabel == labels(i) %Verifying
        pos = pos + 1;
    end
end

t = toc;
fprintf('Testing time: %d min %d sec.\n', floor(t / 60), ...
    round(t - floor(t / 60) * 60));
fprintf('Accuracy: %.2f%%\n', 100 * pos / size(images, 3));
