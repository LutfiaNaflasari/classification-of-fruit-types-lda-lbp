clc; clear;
close all;
warning off all;

folder_name = 'data_latih';
nama_file = dir(fullfile(folder_name,'*.jpg'));
jumlah_file = numel(nama_file);

data_latih = zeros(jumlah_file,61);
kelas_latih = cell(jumlah_file,1);

for n = 1:jumlah_file
    Img = imread(fullfile(folder_name,nama_file(n).name));
    cform = makecform('srgb2lab');
    lab = applycform(Img,cform);
    a = lab(:,:,2);
    bw = a>150;
    bw = imfill(bw,'holes');
    hsv = rgb2hsv(Img);
    grey = rgb2gray(Img);
    h = hsv(:,:,1); 
    s = hsv(:,:,2); 
    h(~bw) = 0;
    s(~bw) = 0;
    data_latih(n,1) = sum(sum(h))/sum(sum(bw));
    data_latih(n,2) = sum(sum(s))/sum(sum(bw));
    lbpimage = extractLBPFeatures(grey);
    data_latih(n,3 : 61) = lbpimage;
end

for h=1:41
    kelas_latih{h} = 'anggur';
end

for h=42:71
    kelas_latih{h} = 'apel';
end

for h=72:101
    kelas_latih{h} = 'jeruk';
end

for h=102:143
    kelas_latih{h} = 'pisang';
end

Mdl = fitcdiscr(data_latih,kelas_latih);
kelas_keluaran = predict(Mdl,data_latih);
jumlah_benar = 0;
for d = 1:jumlah_file
    if isequal(kelas_keluaran{d},kelas_latih{d})
        jumlah_benar = jumlah_benar+1;
    end
end

akurasi_pelatihan = jumlah_benar/jumlah_file*100;

save Mdl Mdl
