function [name] = seam_cut(bin, img, dst)
    global w h;
    name =0; 
    % CC = bwconncomp(image_bw);
    rects = getconcomp(bin); 
%    figure(1), 
%    imshow(img); 
     
%for i = 1:size(rects, 1)  
%       rectangle('position', rects(i, :), 'EdgeColor', 'r'); 
%end
     imwrite( bin, '1261.jpg');

   for i = 1:size(rects, 1)        
       bb = rects(i,:);
       if ~should_save(bb)
           continue;
       end
       %rectangle('position', rects(i, :), 'EdgeColor', 'r'); 
       roi = bin(bb(2):bb(2)+bb(4)-1,bb(1):bb(1)+bb(3)-1);
       roi_rgb = img(bb(2):bb(2)+bb(4)-1,bb(1):bb(1)+bb(3)-1, :);
%       if should_cut(roi)
           name = cut(roi,roi_rgb, dst,name);
%       else
           imwrite(roi_rgb, [dst,'_',num2str(name),'.bmp']);
           [ht, wt] = size(roi);
           h = h + ht;
           w = w + wt;
           name = name+1;
%       end   
    end
end

function flag = should_cut(roi)
    [h,w] = size(roi);
    if w>60 || (w>h*2.1 && w>50) || (w>h*2.3 && w>40) 
        flag = 1;
    else
        flag = 0;
    end
end

% function flag = should_cut(roi)
%     [h,w] = size(roi);
%     if w>120 || (w> h*2 && w>50) || (w>h*2.3 && w>40) 
%         flag = 1;
%     else
%         flag = 0;
%     end
% end

function rects = getconcomp(mat)
    image_bw = ~mat;
    %L = bwlabel(image_bw);%�����ͨ����
    stats = regionprops(image_bw);
    rects = ceil(cat(1,  stats.BoundingBox));
end

function out = merge_rect(rects)
    if(size(rects,1) == 1)
        out = rects;
    else
        out(1) = min(rects(:,1));
        out(2) = min(rects(:,2));
        out(3) = max(rects(:,1) + rects(:,3));
        out(4) = max(rects(:,2) + rects(:,4));
        out(3) = out(3) - out(1);
        out(4) = out(4) - out(2);
    end
end
function save = should_save(bb)
    if(bb(4)<30 || bb(3)<40 || bb(4)*bb(3)<420 || bb(4) > 2.3 * bb(3) || bb(3) > 2.3 * bb(4) )
    %if(bb(4)<30 || bb(3)<40 || bb(4)*bb(3)<420  )
        save = 0;
    else
        save = 1;
    end
end

% function save = should_save(bb)
%     if(bb(4)<20 || bb(3)<20 || bb(4)*bb(3)<700)
%         save = 0;
%     else
%         save = 1;
%     end
% end
function name = cut(roi,roi_rgb, dst, name)
    global w h;
    weight = sum(roi);
    b = 18;
    e = min(45,size(roi,2)-10);
    [~,id] = max(weight(b:e)); 
    id = id + b - 1;

    com1 = roi(:, 1:id);
    com1_rgb = roi_rgb(:, 1:id, : );
    bb = getconcomp(com1);
    if(size(bb ,1 ) ~= 1)
       bb = merge_rect(bb);       
    end
    com1 = com1(bb(2):bb(2)+bb(4)-1,bb(1):bb(1)+bb(3)-1);
    com1_rgb = com1_rgb(bb(2):bb(2)+bb(4)-1,bb(1):bb(1)+bb(3)-1, :);
     
    if(should_save(bb))
        imwrite(com1_rgb, [dst,'_',num2str(name),'.bmp']); 
        name = name+1; 
        [ht, wt] = size(com1);
        h = h + ht;
        w = w + wt;
    end

    com2 = roi(:,id:end);
    com2_rgb = roi_rgb(:,id:end,:);
    bb = getconcomp(com2);
    if(size(bb ,1 ) ~= 1)
       bb = merge_rect(bb);       
    end  
    if(should_save(bb))
        com2 = com2(bb(2):bb(2)+bb(4)-1,bb(1):bb(1)+bb(3)-1);
        com2_rgb = com2_rgb(bb(2):bb(2)+bb(4)-1,bb(1):bb(1)+bb(3)-1, :);
        if should_cut(com2)
            name = cut(com2,com2_rgb, dst,name);
        else
            imwrite(com2_rgb, [dst,'_',num2str(name),'.bmp']); 
            name = name+1;
            [ht, wt] = size(com1);
            h = h + ht;
            w = w + wt;
        end 
    end
end

% Ar = cat(1, stats.Area);
% ind = find(Ar ==max(Ar));%�ҵ������ͨ����ı��
% image_bw(find(L~=ind))=0;%������������Ϊ0
% figure,imshow(image_bw);%��ʾ�����ͨ����



% close all
% clc
% bw=imread('3.jpg');
% bw=medfilt2(bw);
% planes=bwareaopen(bw,50);
% D=mat2gray(bwdist(imcomplement(planes)));
% stats=regionprops(D>.8,'Centroid');
% planes_centroid=cat(1,stats.Centroid);
% planes_mask=false(size(bw));
% planes_mask(sub2ind(size(bw),round(planes_centroid(:,2)),round(planes_centroid(:,1))))=1;;
% M=imimposemin(imcomplement(D),planes_mask);
% L=watershed(M);
% r=L & planes;
% stats=regionprops(r,'BoundingBox','Centroid');
% bb=cat(1,stats.BoundingBox);
% c=cat(1,stats.Centroid);
% figure,imshow(planes)
% hold on
% for i=1:length(stats)
%     rectangle('Position',bb(i,:),'EdgeColor','b')
%     plot(c(i,1),c(i,2),'r*')
%     text(c(i,1)-5,c(i,2)-10,num2str(i))
% end
