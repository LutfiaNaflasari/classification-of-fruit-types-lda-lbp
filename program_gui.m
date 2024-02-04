function varargout = program_gui(varargin)
% PROGRAM_GUI MATLAB code for program_gui.fig
%      PROGRAM_GUI, by itself, creates a new PROGRAM_GUI or raises the existing
%      singleton*.
%
%      H = PROGRAM_GUI returns the handle to a new PROGRAM_GUI or the handle to
%      the existing singleton*.
%
%      PROGRAM_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PROGRAM_GUI.M with the given input arguments.
%
%      PROGRAM_GUI('Property','Value',...) creates a new PROGRAM_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before program_gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to program_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help program_gui

% Last Modified by GUIDE v2.5 19-Nov-2022 18:45:47

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @program_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @program_gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before program_gui is made visible.
function program_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to program_gui (see VARARGIN)

% Choose default command line output for program_gui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
movegui(hObject, "center");

% UIWAIT makes program_gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = program_gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[nama_file, nama_folder] = uigetfile('*.jpg');
if ~isequal(nama_file,0)
    Img = imread(fullfile(nama_folder,nama_file));
    axes(handles.axes1)
    imshow(Img)
    title('Citra RGB')
    set(handles.edit1,'String',nama_file)
    handles.Img = Img;
    guidata(hObject, handles)
    
else
    return
end

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

Img = handles.Img;
cform = makecform('srgb2lab');
lab = applycform(Img,cform);
a = lab(:,:,2);
bw = a>140;
bw = imfill(bw,'holes');
grey = rgb2gray(Img);
axes(handles.axes2)
imshow(bw)
title('Citra Biner')
axes(handles.axes3)
imshow(lab)
title('Citra L*a*b')
handles.bw = bw;
guidata(hObject, handles)

handles.grey = grey;
guidata(hObject, handles)

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

Img = handles.Img;
bw = handles.bw;
grey = handles.grey;
hsv = rgb2hsv(Img);
h = hsv(:,:,1); 
s = hsv(:,:,2); 
h(~bw) = 0;
s(~bw) = 0;
lbpimage = extractLBPFeatures(grey);
data_uji(1,1) = sum(sum(h))/sum(sum(bw));
data_uji(1,2) = sum(sum(s))/sum(sum(bw));
data_uji(1,3 : 61) = lbpimage;
data_tabel = cell(3,2);
data_tabel{1,1} = 'Hue';
data_tabel{2,1} = 'Saturation';
data_tabel{3,1} = 'LBP';
data_tabel{1,2} = num2str(data_uji(1,1));
data_tabel{2,2} = num2str(data_uji(1,2));
for i = 2 : 60
    data_tabel{3,i} = num2str(data_uji(1,i-1));
end
set(handles.uitable1,'Data',data_tabel,'RowName',1:3)
axes(handles.axes4);
cla('reset');
h = histogram(lbpimage);
grid on;
title('Histogram LBP');
handles.data_uji = data_uji;
guidata(hObject, handles)

% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

data_uji = handles.data_uji;
load Mdl;
kelas_keluaran = predict(Mdl,data_uji);
% jumlah_benar = 0;
% for k = 1:jumlah_file
%     if isequal(kelas_keluaran{k},kelas_uji{k})
%         jumlah_benar = jumlah_benar+1;
%     end
% end
% akurasi_pengujian = jumlah_benar/jumlah_file*100;
set(handles.edit2,'String',kelas_keluaran{1});
% set(handles.edit3,'String', akurasi_pengujian);

% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(handles.edit1,'String',[])
set(handles.edit2,'String',[])
% set(handles.edit3,'String',[])


axes(handles.axes1)
cla reset
set(gca,'XTick',[])
set(gca,'YTick',[])

axes(handles.axes2)
cla reset
set(gca,'XTick',[])
set(gca,'YTick',[])

axes(handles.axes3)
cla reset
set(gca,'XTick',[])
set(gca,'YTick',[])

axes(handles.axes4)
cla reset
set(gca,'XTick',[])
set(gca,'YTick',[])

set(handles.uitable1,'Data',[],'RowName',{'' '' '' ''})

function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
