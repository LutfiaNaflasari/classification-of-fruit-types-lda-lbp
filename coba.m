clc; clear;
close all;
warning off all;

nama_folder = 'data_coba';
nama_file = dir(fullfile(nama_folder, '*.jpg'));
jumlah_file = numel(nama_file);

data_uji = zeros(jumlah_file,61);
kelas_uji = cell(jumlah_file,1);

for n = 1:jumlah_file
    Img = imread(fullfile(nama_folder,nama_file(n).name));
    cform = makecform('srgb2lab');
    lab = applycform(Img,cform);
    a = lab(:,:,2);
    bw = a>140;
    bw = imfill(bw,'holes');
    hsv = rgb2hsv(Img);
    grey = rgb2gray(Img);
    h = hsv(:,:,1); 
    s = hsv(:,:,2); 
    h(~bw) = 0;
    s(~bw) = 0;
    data_uji(n,1) = sum(sum(h))/sum(sum(bw));
    data_uji(n,2) = sum(sum(s))/sum(sum(bw));
    lbpimage = extractLBPFeatures(grey);
    data_uji(n,3 : 61) = lbpimage;
end

% for k=1:5
%     kelas_uji{k} = 'anggur';
% end
% 
% for k=6:21
%     kelas_uji{k} = 'apel';
% end
% 
% for k=22:40
%     kelas_uji{k} = 'jeruk';
% end
% 
% for k=41:45
%     kelas_uji{k} = 'pisang';
% end

for h=1:30
    kelas_uji{h} = 'anggur';
end

for h=31:60
    kelas_uji{h} = 'apel';
end

for h=61:90
    kelas_uji{h} = 'jeruk';
end

for h=91:120
    kelas_uji{h} = 'pisang';
end

load Mdl

kelas_keluaran = predict(Mdl,data_uji);

jumlah_benar = 0;
for k = 1:jumlah_file
    if isequal(kelas_keluaran{k},kelas_uji{k})
        jumlah_benar = jumlah_benar+1;
    end
end

akurasi_pengujian = jumlah_benar/jumlah_file*100;