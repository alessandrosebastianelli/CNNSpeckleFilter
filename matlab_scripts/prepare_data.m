%% Set the root path
%root = '/Volumes/ELEMENTS/Dati_CSG/San_Francisco_1/';
root = '/media/sebbyraft/phd/Dati_CSG/San_Francisco_1/';

%% Load image paths
root2 = strcat(root, 'Sottobande/');
filesTree = dir(root);
filesTree2 = dir(root2);

paths = table();
paths.images = {};

disp('Loading paths...');

for i=1:length(filesTree)
    file = filesTree(i).name;
    if contains(file, 'jpg')
        img = imread(strcat(root, filesTree(i).name));
    elseif contains(file, '.') == false && contains(file, '_') == true
        paths.images(end+1) = {strcat(root, filesTree(i).name)};
    end
end

for i=1:length(filesTree2)
    file = filesTree2(i).name;
    if contains(file, '.') == false && contains(file, '_') == true
        paths.images(end+1) = {strcat(root2, filesTree2(i).name)};
    end
end
%% Save images as TIF file
s = size(img);
X1 = s(1);
X2 = s(2);

saveName = split(root,"/");
saveName = saveName(end-1);
savePath = strcat('/media/sebbyraft/phd/', saveName{1});
mkdir(savePath)

for i=1:height(paths)
    disp(strcat('Converting image-', num2str(i), '-of-', num2str(height(paths))));
    path = paths.images(i);
    TT1 = leggi(path{1},X1,X2,[],1);
    %TT2 = TT1/max(max(TT1));
    %TT3 = normalize(TT2);
    R = real(TT1);
    %R = (R - min(R(:)))/(max(R(:)) - min((R(:))));
    I = imag(TT1);
    %I = (I - min(I(:)))/(max(I(:)) - min((I(:))));
    
    saveastiff(single(R), strcat(savePath,'/',saveName{1},'_',num2str((2*(i-1) + 1)),'.tiff'));
    saveastiff(single(R), strcat(savePath,'/',saveName{1},'_',num2str(2*i),'.tiff'));

    %imwrite(R,strcat(savePath,'/',saveName{1},'_',num2str((2*(i-1) + 1)),'.tiff'), 'tiff'); 
    %imwrite(I,strcat(savePath,'/',saveName{1},'_',num2str(2*i),'.tiff'), 'tiff'); 
end