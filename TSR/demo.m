clear
addpath common/
addpath common/fminlbfgs
addpath Functions/

%% STEP 0: Initialization

imageChannels = 3;                                   % number of channels (rgb, so 3)

patchDim   = 8;                                      % patch dimension
% numPatches = 7292;%24*24*3;                                 % number of patches

visibleSize = patchDim * patchDim * imageChannels;   % number of input units 
outputSize  = visibleSize;                           % number of output units

sparsityParam = 0.035;                               % desired average activation of the hidden units.
lambda = 3e-3;                                       % weight decay parameter       
beta = 5;                                            % weight of sparsity penalty term       

epsilon = 0.1;                                       % epsilon for ZCA whitening
% Blocks = 9;
Wchoices = [3,4,5,6,7,8,9,10,11];

%% STEP 2a: Load patches
workspace = load('../RGB/FeaturesPatches/Stride8/featuresTrainOriginal.mat');
trainPatches = im2double(workspace.trainFeatures);
trainLabels = workspace.trainLabels;
numTrainImgs = size(trainPatches,2);

Blocks = size(trainPatches,1)/192;

%% STEP 2b: Apply preprocessing

numZ = 1;
ZCAWhite = cell(numZ);
meanPatch = cell(numZ);

AllPatches = [];
for ii = 1:Blocks
AllPatches = [AllPatches,trainPatches((ii-1)*192+1:(ii*192),:)];
end

for ii = 1:numZ
    % Subtract mean patch (hence zeroing the mean of the patches)
    %meanEachPatch = mean(patches);
    meanPatch = mean(AllPatches,2);
    AllPatches = bsxfun(@minus, AllPatches, meanPatch);
    
    % Apply ZCA whitening
    sigma = AllPatches * AllPatches' / length(AllPatches);
    [u, s, v] = svd(sigma);
    ZCAWhite = u * diag(1 ./ sqrt(diag(s) + epsilon)) * u';
    AllPatches = ZCAWhite * AllPatches;
end

trainPatches = [];
for ii = 1:Blocks
trainPatches = [trainPatches;AllPatches(:,(ii-1)*numTrainImgs+1:(ii*numTrainImgs))];
end

%% Train
% % % workspace = load('WeightsPatches/CURE-TSR_Stride8.mat');
% workspace = load('WeightsPatches/ImageNet_8x8_Weights_L2.mat');
% W = workspace.W;
% b = workspace.b;

workspace = load('WeightsPatches/ClassifiedWeights.mat');
Wall = workspace.Wnew;
ball = workspace.bnew;

W = [];
b = [];
for ii = 1:length(Wchoices)
    W = [W;Wall{Wchoices(ii)}];
    temp = ball{Wchoices(ii)};
    b = [b;temp(:)];
end

for numBlocks = 1:Blocks
    
[temp_FirstLayer] = feedForwardAE(W,b, trainPatches(visibleSize*(numBlocks-1)+1:visibleSize*numBlocks,:));
trainFeatures(hiddenChoices*(numBlocks-1)+1:hiddenChoices*numBlocks,:) = temp_FirstLayer;          

end 

% % Prepeocessing
trainFeatures = bsxfun(@minus, trainFeatures, (mean(trainFeatures)));
trainFeatures = normc(trainFeatures);

inputDataOriginal = trainFeatures;

%% ----------------- YOUR CODE HERE ----------------------

lambda = 1e-4;
numClasses = numel(unique(trainLabels));

options = struct;
options.HessUpdate = 'lbfgs';
options.MaxIter = 400;
options.Display = 'iter';
options.GradObj = 'on';

softmaxModel = softmaxTrain(hiddenChoices*Blocks, numClasses, lambda, ...
    trainFeatures, trainLabels, options);

[trainPred] = softmaxPredict(softmaxModel, trainFeatures);
acc = mean(trainLabels(:) == trainPred(:));
fprintf('Training acc is %f%%\n', 100*acc);

%% Test
Present = {'Decolorization'};


accuracy = zeros(12,5);
predAll = [];
labelsAll = [];

workspace = load('../RGB/FeaturesPatches/Stride8/featuresTestOriginal.mat');
testPatches = im2double(workspace.testFeatures);
testLabels = workspace.testLabels;
labelsAll = [labelsAll;testLabels(:)];
numTestImgs = size(testPatches,2);

%% Preprocessing

ZCAWhite = cell(numZ);
meanPatch = cell(numZ);

AllPatches = [];
for ii = 1:Blocks
    AllPatches = [AllPatches,testPatches((ii-1)*192+1:(ii*192),:)];
end

for ii = 1:numZ
    % Subtract mean patch (hence zeroing the mean of the patches)
    %meanEachPatch = mean(patches);
    meanPatch = mean(AllPatches,2);
    AllPatches = bsxfun(@minus, AllPatches, meanPatch);
    
    % Apply ZCA whitening
    sigma = AllPatches * AllPatches' / length(AllPatches);
    [u, s, v] = svd(sigma);
    ZCAWhite = u * diag(1 ./ sqrt(diag(s) + epsilon)) * u';
    AllPatches = ZCAWhite * AllPatches;
end

testPatches = [];
for ii = 1:Blocks
    testPatches = [testPatches;AllPatches(:,(ii-1)*numTestImgs+1:(ii*numTestImgs))];
end

for numBlocks = 1:Blocks
    
    temp_FirstLayer = feedForwardAE(W, b, testPatches(visibleSize*(numBlocks-1)+1:visibleSize*numBlocks,:));
    testFeatures(hiddenChoices*(numBlocks-1)+1:hiddenChoices*numBlocks,:) = temp_FirstLayer;
    
end

% % Prepeocessing
testFeatures = bsxfun(@minus, testFeatures, (mean(testFeatures)));
testFeatures = normc(testFeatures);

[pred] = softmaxPredict(softmaxModel, testFeatures);
acc = mean(testLabels(:) == pred(:));
fprintf('Original Test is %f%%\n', 100*mean(pred(:) == testLabels(:)));
accuracy(:,1) = acc;

for cT = 1:length(Present)
    
    for cL = 1:5

        workspace = load(['../RGB/FeaturesPatches/Stride8/',Present{cT},'/featuresTest',Present{cT},sprintf('%d.mat',cL)]);
        testPatches = im2double(workspace.testFeatures);
        testLabels = workspace.testLabels;
        
        ZCAWhite = cell(numZ);
        meanPatch = cell(numZ);
        
        AllPatches = [];
        for ii = 1:Blocks
            AllPatches = [AllPatches,testPatches((ii-1)*192+1:(ii*192),:)];
        end
        
        for ii = 1:numZ
            % Subtract mean patch (hence zeroing the mean of the patches)
            %meanEachPatch = mean(patches);
            meanPatch = mean(AllPatches,2);
            AllPatches = bsxfun(@minus, AllPatches, meanPatch);
            
            % Apply ZCA whitening
            sigma = AllPatches * AllPatches' / length(AllPatches);
            [u, s, v] = svd(sigma);
            ZCAWhite = u * diag(1 ./ sqrt(diag(s) + epsilon)) * u';
            AllPatches = ZCAWhite * AllPatches;
        end
        
        testPatches = [];
        for ii = 1:Blocks
            testPatches = [testPatches;AllPatches(:,(ii-1)*numTestImgs+1:(ii*numTestImgs))];
        end
        
        
        for numBlocks = 1:Blocks
            
            temp_FirstLayer = feedForwardAE(W, b, testPatches(visibleSize*(numBlocks-1)+1:visibleSize*numBlocks,:));
            testFeatures(hiddenChoices*(numBlocks-1)+1:hiddenChoices*numBlocks,:) = temp_FirstLayer;
            
        end
        
        % % Prepeocessing
        testFeatures = bsxfun(@minus, testFeatures, (mean(testFeatures)));
        testFeatures = normc(testFeatures);
        
        [pred] = softmaxPredict(softmaxModel, testFeatures);
        predAll = [predAll;pred(:)];
        
        acc = mean(testLabels(:) == pred(:));
        fprintf([Present{cT},sprintf(' lvl %d : %0.3f\n',cL, acc * 100)]);
        
        accuracy(cT,(cL+1)) = acc;
        
    end
end