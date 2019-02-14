function [Features,Labels] = extractRGB(folderPath)

Labels = [];
count = 1;
imsize = 28;

for sgnType = 1:14
    
    folder = sprintf('%02d',sgnType);
    path = [folderPath,num2str(folder),'/*.bmp'];
    imagefiles = dir(path);
    nfiles = length(imagefiles);    % Number of files found
    
    for ii=1:nfiles
        
        currentfilename = imagefiles(ii).name;
        currentImage = im2double(imread(currentfilename));
        
        R = imresize(currentImage(:,:,1),[imsize,imsize]);
        G = imresize(currentImage(:,:,2),[imsize,imsize]);
        B = imresize(currentImage(:,:,3),[imsize,imsize]); 
         
        currentImage = cat(3,R,G,B);
        Features(:,count) = currentImage(:);
        
        Labels = [Labels;sgnType];
        count = count+1;
        
    end
    
end
numFeatures = length(Features);
Ind = randperm(numFeatures,numFeatures);
Features = Features(:,Ind');
Labels = Labels(Ind');

end