function varargout = audiovisual_acquire(varargin)
% AUDIOVISUAL_ACQUIRE MATLAB code for audiovisual_acquire.fig
%      AUDIOVISUAL_ACQUIRE, by itself, creates a new AUDIOVISUAL_ACQUIRE or raises the existing
%      singleton*.
%
%      H = AUDIOVISUAL_ACQUIRE returns the handle to a new AUDIOVISUAL_ACQUIRE or the handle to
%      the existing singleton*.
%
%      AUDIOVISUAL_ACQUIRE('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in AUDIOVISUAL_ACQUIRE.M with the given input arguments.
%
%      AUDIOVISUAL_ACQUIRE('Property','Value',...) creates a new AUDIOVISUAL_ACQUIRE or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before audiovisual_acquire_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to audiovisual_acquire_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help audiovisual_acquire

% Last Modified by GUIDE v2.5 03-Jul-2015 18:47:52

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @audiovisual_acquire_OpeningFcn, ...
    'gui_OutputFcn',  @audiovisual_acquire_OutputFcn, ...
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

% --- Executes just before audiovisual_acquire is made visible.
function audiovisual_acquire_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to audiovisual_acquire (see VARARGIN)

% Choose default command line output for audiovisual_acquire
handles.output = hObject;
handles.cnt = 1;

% Audio cue properties
freq = 2e3; handles.samp = 44100; % Beep and sampling frequncies (Hz)
handles.beep_dur = 0.4; pts = round(handles.beep_dur*handles.samp); % Beep duration (sec);
handles.audiolat = 0.4; % Estimated sound card latency
handles.jaerlat = 3.2; % Estimated jAER logging latency
svect = sin(2*pi*freq*linspace(0,pts-1,pts)/handles.samp); % Audio vector
handles.player = audioplayer(svect,handles.samp);
handles.recorder = audiorecorder(handles.samp,16,2); % 16-bits, stereo

%open udp port, u1 for cochlea, u2 for retina
handles.u1=udp('localhost',8997); % make a UDP interface to localhost on port 8995 (the default port for AEViewer RemoteControl interface)
fopen(handles.u1);
handles.u2=udp('localhost',8998); % make a UDP interface to localhost on port 8997 (the default port for AEViewer RemoteControl interface)
fopen(handles.u2);

% Create timer function
handles.timer = timer(...
    'ExecutionMode', 'fixedRate', ...   % Run timer repeatedly
    'Period', 1, ...         % Initial period is timer_period sec.
    'TimerFcn', {@update_display,hObject}); % Specify callbackhandles.timer = timer(...
handles.start = 0; % Not started yet

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes audiovisual_acquire wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --- Outputs from this function are returned to the command line.
function varargout = audiovisual_acquire_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes during object creation, after setting all properties.
function text2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

set(hObject,'String','Welcome');


% --- Executes during object creation, after setting all properties.
function text3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

set(hObject,'String','');


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Read list of words
handles.filname = get(handles.edit4,'String');
fid = fopen(handles.filname);
C = textscan(fid, '%s'); wlist = C{1}; nwords = size(wlist);
fclose(fid);
handles.wlist = wlist; handles.nwords = nwords;

handles.rec_dur = str2num(get(handles.edit1,'String'));
timer_period = handles.rec_dur + handles.beep_dur + handles.audiolat + ...
    handles.jaerlat; % sec
set(handles.timer, 'Period', timer_period);   % Set timer period

handles.sid = get(handles.edit2,'String'); % Subject ID
handles.tid = get(handles.edit5,'String'); % Trial ID
mkdir([handles.sid filesep handles.tid]);

% Start timer
if strcmp(get(handles.timer, 'Running'), 'off')
    start(handles.timer);
end

handles.start = 1; % Started

% Update handles structure
guidata(hObject, handles);

% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Necessary to provide this function to prevent timer callback
% from causing an error after code stops executing.
% Before exiting, if the timer is running, stop it.
if strcmp(get(handles.timer, 'Running'), 'on')
    stop(handles.timer);
end
% Destroy timer
delete(handles.timer)

% Clean up udp connection
fclose(handles.u1); % clean up the UDP connection
delete(handles.u1);
fclose(handles.u2); % clean up the UDP connection
delete(handles.u2);

% Hint: delete(hObject) closes the figure
delete(hObject);

% START USER CODE
function update_display(hObject, eventdata, hfigure)
% Timer timer1 callback, called each time timer iterates.

coch_suffix = '_coch'; ret_suffix = '_ret';
handles = guidata(hfigure);
cnt = handles.cnt;

% Display next word
if handles.start
    wlist = handles.wlist; nwords = handles.nwords;
    if cnt < nwords(1)+1
        handles.rec_dur
        %jaer command
         commandJAER(handles.u1,['startlogging ' pwd filesep handles.sid ...
             filesep handles.tid filesep wlist{cnt} coch_suffix])
         commandJAER(handles.u2,['startlogging ' pwd filesep handles.sid ...
             filesep handles.tid filesep wlist{cnt} ret_suffix])
        
        set(handles.uipanel1,'BackgroundColor','Red');
        set(handles.text2,'String','');
        
        playblocking(handles.player);
        set(handles.uipanel1,'BackgroundColor','Green');
        set(handles.text2,'String',wlist{cnt});
        set(handles.text3,'String',num2str(cnt));
        %mark
         recordblocking(handles.recorder,handles.rec_dur); pause(handles.rec_dur);
         out = getaudiodata(handles.recorder);
         audiowrite([handles.sid filesep handles.tid filesep wlist{cnt} '.wav' ],out,handles.samp);
        
        handles.cnt = cnt+1;
        
        %jaer command
         commandJAER(handles.u1,'stoplogging')
         commandJAER(handles.u2,'stoplogging')
    end
    if cnt == nwords(1)+1
        set(handles.text2,'String','Finished');
    end
end

% Update handles structure
guidata(hfigure, handles);
% END USER CODE


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



function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
