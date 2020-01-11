%TRAIN performs training of your network
%
%   Sergey Shuvaev, 2016. sshuvaev@cshl.edu

close all
clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Your learning rate parameters here %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

LRINIT = 0.01; %Initial learning rate
LRDECAY = 0.01; %Overall decrease of the learning rate throughout training

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Your network architecture here %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     #                 TYPE        INDIM       OUTDIM      WDIM        NLTYPE      OS
leNet(1)=DLNetworkLayer('input',    [32 32 1],  [32 32 1],  [],         [],         []);
leNet(2)=DLNetworkLayer('conv',     [32 32 1],  [28 28 6],  [5 5 1 6],  'sigmoid',  []);
leNet(3)=DLNetworkLayer('maxpool',  [28 28 6],  [14 14 6],  [2 2],      [],         []);
leNet(4)=DLNetworkLayer('conv',     [14 14 6],  [10 10 16], [5 5 6 16], 'sigmoid',  []);
leNet(5)=DLNetworkLayer('maxpool',  [10 10 16], [5 5 16],   [2 2],      [],         []);
leNet(6)=DLNetworkLayer('full',     [5 5 16],   [120 1 1],  [120 400],  'sigmoid',  []);
leNet(7)=DLNetworkLayer('full',     [120 1 1],  [84 1 1],   [84 120],   'sigmoid',  []);
leNet(8)=DLNetworkLayer('full',     [84 1 1],   [10 1 1],   [10 84],    'sigmoid',  []);
leNet(9)=DLNetworkLayer('target',   [10 1 1],   [10 1 1],   [],         [],         []);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Your images and labels here %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~exist('mnist', 'dir') %Download MNIST dataset only once
    disp('Downloading MNIST data set...');
    gunzip('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'mnist');
    gunzip('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 'mnist');
    gunzip('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 'mnist');
    gunzip('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 'mnist');
end

%Loading
images = loadMNISTImages(fullfile('mnist','train-images-idx3-ubyte'));
labels = loadMNISTLabels(fullfile('mnist','train-labels-idx1-ubyte'));

images_tmp = reshape(images, [28, 28, size(images, 2)]); %Preprocessing
images = zeros(32, 32, size(images, 2));
images(3 : 30, 3 : 30, :) = images_tmp; %Zero padding
images = images * 1.275 - 0.1; %Mean 0, variance 1
clear images_tmp;

%%%%%%%%%%%%
% Training %
%%%%%%%%%%%%

len = length(leNet);
tic;

for i = 1 : size(images, 3) %Learning
    
    fprintf('%d\n', i);
    
    leNet(1).setInput(images(:, :, i)); %Initializing
    leNet(len).setTarget(labels(i));
    
    for j = 2 : len - 1 %Forward prop
        stepForward(leNet, j);
    end
    
    for j = len : - 1 : 2 %Back prop
        stepBackward(leNet, j, LRINIT * LRDECAY ^ (i / size(images, 3)));
    end
end

t = toc;
fprintf('Training time: %d min %d sec.\n', floor(t / 60), ...
    round(t - floor(t / 60) * 60));
