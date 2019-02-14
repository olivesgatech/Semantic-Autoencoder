clear

workspace = load('AccuraciesClassifiedWeights.mat');
accuracyAll = workspace.AccuracySeparate;

workspace = load('accuracy_ImageNet_Stride8.mat');
accuracyCombined = workspace.accuracy;

workspace = load('accuracy_ImageNet_Edges_Stride4.mat');
accuracyEdges = workspace.accuracy;

ind = 1:1:5;
count = 1;
Present = {'Decolorization','LensBlur','CodecError','Darkening'...
    'DirtyLens','Exposure','GaussianBlur','Noise','Rain','Shadow','Snow','Haze'};

figure(1)
while (count<13)
    subplot(4,3,count)
    for ii = 1:3%length(accuracyAll)
        if ii == 1
            PresAcc = accuracyAll{ii};
        elseif ii == 2
            PresAcc = accuracyCombined;
        else 
            PresAcc = accuracyEdges;
        end
        plot(ind, PresAcc(count,:));
        hold on
        axis([1 6 0.2 1])
%         legend('Z1','Z2','Z3','Z4','Z5');%'AE','Original','ES','WP','Original900','Original&Null')
    end
    legend('Co','All','Edg');%,'NC','LT','LB','All');%'RT','RB','All');%,'T','B','L','R');%'AE','Original','ES','WP','Original900','Original&Null')
    title(Present{count})
    count = count+1;
end
