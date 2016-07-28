clc;
clear;
close all;
% src = 'E:\Write_identification\dataset\icdar2013\experimental_dataset_2013';
% dstTrain = 'E:\Write_identification\dataset\icdar2013\verticalCut';

% src = 'E:\Write_identification\dataset\icdar2013\icdar2013_benchmarking_dataset';
% dstTrain = 'E:\Write_identification\dataset\icdar2013\icdar2013_benchmarking_verticalCut';

src = '../python/meitu_train';
dstTrain = '../python/meitu_train_cut';

if  exist(dstTrain, 'dir')
    rmdir( dstTrain,'s' );
end

imgs = dir( fullfile( src, '*.jpg' ) );
len=size(imgs,1);
count  = 0;
global w h;
w= 0; h = 0;
thres = 50;
for i=1:len    
    fprintf('processing %dth image\n', i);
    nm = imgs(i).name;    
    I = imread( fullfile(src, nm ) );
    level = graythresh(I)
    %level = min(0.4, level)
    binary = im2bw( I, level );
%    gray = rgb2gray(I);
%    binary = im2bw(gray,0.4);
%     figure(1);
%     imshow( binary)
%    Y=ordfilt2(binary,9,ones(3,3));
%     figure(2);
%     imshow(Y);
%    power = sum( ~Y,2);
%    id = find(diff(power > thres)) ;
%    idx = find( diff(id) > 400 );
%    idd = id( idx + 1 );
%    if length( idd) == 0
%        a = 1;b = size( Y,1);
%    elseif length ( idd) == 1 
%        a = idd(1); b = size( Y,1);
%    else
%       a = idd(1) ;b = idd(2);    
%    end
%    
%            bin = Y(1:b,:);
%            I = I(1:b,:,:);
%    
% 
%    figure(3);
%    imshow( bin);
    
    dst = fullfile(dstTrain, nm(1:5));
    if ~exist(dst, 'dir')
        mkdir(dst);
    end
    dst = fullfile(dst, nm(1:5));
    count = count + seam_cut(binary,I,dst);
    close;
end
aw = w/count;
ah = h/count;

fp = fopen(fullfile(dstTrain,'readme.txt'),'w');
fprintf(fp,'total patch:%d\n',count);
fprintf(fp,'average width  of patch:%f\n',aw);
fprintf(fp,'average height of patch:%f\n',ah);
fclose(fp);

