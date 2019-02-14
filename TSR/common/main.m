clear

%% Train

trainFolderPath = '../../Train/Real/Original/';
[trainFeatures,trainLabels] = extractRGB(trainFolderPath);
save('Features/featuresTrainOriginal.mat','trainFeatures','trainLabels');
% load featuresTrainVirtualColor.mat

%% Test Original

% Original
testFolderPath = '../Real/Original/';
[testFeatures,testLabels] = extractRGB(testFolderPath);
save('Features/featuresTestOriginal.mat','testFeatures','testLabels');
% load featuresTestVirtualColor.mat

clear testFeatures
clear testLabels

%% Test Challenges

Present = {'Decolorization','LensBlur','CodecError','Darkening',...
    'DirtyLens','Exposure','GaussianBlur','Noise','Rain','Shadow','Snow','Haze'};

for cT = 1:length(Present)
    
    cT
    for cL = 1:5
        
        testFolderPath = ['../Real/',Present{cT},'/',Present{cT},sprintf('-%d/',cL)];
        [testFeatures,testLabels] = extractRGB(testFolderPath);
        pathImageSave = ['Features/',Present{cT},'/'];
        
        if ~exist(pathImageSave, 'dir')
            mkdir(pathImageSave);
        end
        
        save(['Features/',Present{cT},'/featuresTest',Present{cT},sprintf('%d.mat',cL)],'testFeatures','testLabels');
        
        clear testFeatures
        clear testLabels
        
    end
    
end