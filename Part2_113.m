close all;
clear all;
clc;
Load the sound track
[X, Fs] = audioread('inception_sound_track.wav');
input=audioplayer(X, Fs);
play(input);
X(:,2) = []; %select one  chanel
origmag=abs(fft(X)); % taking magnitude of input
n =352798 ;
R = [0 1];
phase=2*pi*rand(n,1)*range(R)+min(R); %random pahse
iter=100;
newft=origmag.*phase;
newsignal=ifft(newft);
for i= 1:iter
    newft=origmag.*phase;
    newsignal=ifft(newft);
    newsignal=real(newsignal);
    newft=fft(newsignal);
    phase=angle(newft);
end 
outputsignal=ifft(origmag.*phase); %extracting reconstructed signal
outputsignal=abs(outputsignal); %taking real part of outoutsignal with phase included
outputreal=audioplayer(outputsignal, Fs);
%play(outputreal);
audiowrite("fftoutput.wav", outputsignal,Fs) %saving as wav file
#ATTEMPTING TO FIND THE BEST WINDOW
close all;
clear all;
clc;
% Load the sound track
[X, Fs] = audioread('inception_sound_track.wav');
input=audioplayer(X, Fs);
play(input);
X(:,2) = [];%using one chanel of the input
%lots of windows to test out
windows5000=[flattopwin(5000),parzenwin(5000),bohmanwin(5000),blackmanharris(5000),nuttallwin(5000),barthannwin(5000),rectwin(5000),triang(5000),blackman(5000),hamming(5000),hann(5000),bartlett(5000)];

windows2000=[flattopwin(2000),parzenwin(2000),bohmanwin(2000),blackmanharris(2000),nuttallwin(2000),barthannwin(2000),rectwin(2000),triang(2000),blackman(2000),hamming(2000),hann(2000),bartlett(2000)];
windows1000=[flattopwin(1000),parzenwin(1000),bohmanwin(1000),blackmanharris(1000),nuttallwin(1000),barthannwin(1000),rectwin(1000),triang(1000),blackman(1000),hamming(1000),hann(1000),bartlett(1000)];
windows600=[flattopwin(600),parzenwin(600),bohmanwin(600),blackmanharris(600),nuttallwin(600),barthannwin(600),rectwin(600),triang(600),blackman(600),hamming(600),hann(600),bartlett(600)];
windows300=[flattopwin(300),parzenwin(300),bohmanwin(300),blackmanharris(300),nuttallwin(300),barthannwin(300),rectwin(300),triang(300),blackman(300),hamming(300),hann(300),bartlett(300)];
iter=100 %number of iteration
windowsize=5000 %change according to which window size is being test
[m,n]=size(windows5000); 
overlaps=[windowsize*.5, windowsize*.75] % testing overlaps equalivalent to half and 3/4 of the windowsize
[s,t]=size(overlaps) 
for i= 1:n %for loop to go thru all windows 
   for k=  1:t %nested for to go through each overlap size
     [y,z]=size(windows5000(:,i)); %pick window 
     orig_mag=abs(stft(X,'Window', windows5000(:,i), 'OverlapLength',overlaps(k),'FFTLength', m)); %extract mag
     n =windowsize ;
     R = [0 1]; 
     phase=2*pi*rand(n,1)*range(R)+min(R); %create random phase
      for a = 1:iter
        %griffin lim algorithm as described in spec
        new_ft = orig_mag .*exp(1j.*phase); 
        new_signal = istft(new_ft,'Window',windows5000(:,i), 'OverlapLength',overlaps(k),'FFTLength',windowsize);
        new_signal = real(new_signal);
        new_ft = stft(new_signal,'Window',windows5000(:,i), 'OverlapLength',overlaps(k),'FFTLength',windowsize);
        phase = angle(new_ft);
      end
       %obtain final singal through inverse istft
      final_signal = istft(new_ft,'Window',windows5000(:,i), 'OverlapLength',overlaps(k),'FFTLength',windowsize);
      %naming_convention to sort through wav files
      if i==1
        filename=strcat('flattop','_',string(overlaps(k)),'250it', '.wav')
      elseif i==2
        filename=strcat('parzen','_',string(overlaps(k)),'250it', '.wav')
      elseif i==3
           filename=strcat('bohman','_',string(overlaps(k)),'250it', '.wav')
      elseif i==4
          filename=strcat('blackmanharris','_',string(overlaps(k)),'250it', '.wav')
      elseif i==5
          filename=strcat('nutall','_',string(overlaps(k)),'250it', '.wav')
      elseif i==6
          filename=strcat('barthan','_',string(overlaps(k)),'250it', '.wav')
      elseif i==7
          filename=strcat('rect','_',string(overlaps(k)),'250it', '.wav')
      elseif i==8
          filename=strcat('triang','_',string(overlaps(k)),'250it', '.wav')
      elseif i==9
          filename=strcat('blackman','_',string(overlaps(k)),'250it', '.wav')
      elseif i==10
          filename=strcat('hamming','_',string(overlaps(k)),'250it', '.wav')
      elseif i==11
          filename=strcat('hann','_',string(overlaps(k)),'250it', '.wav')
      elseif i==12
          filename=strcat('bartlett','_',string(overlaps(k)),'250it', '.wav')
      end
      audiowrite(filename,abs(final_signal),Fs)
    end
 end 


%best window deemed to be hann window length 2000 
%perform griffin lim algorithm for 1,50,100,250,500 iterations
diff=[]
windowlength=2000;
windowtype=hann(windowlength);
size(windowtype)
fftlength=2000;
overlap=1500;
origmag=abs(stft(X,'Window',windowtype, 'OverlapLength',overlap,'FFTLength', fftlength));
origphase=angle(stft(X,'Window',windowtype, 'OverlapLength',overlap,'FFTLength', fftlength)); %save the orignal phase to calculate frobenius norm
n =2000 ;
R = [0 1];
phase=2*pi*rand(n,1)*range(R)+min(R);
iter=[1, 50, 100, 250, 500];
[y,z]=size(iter);
newft=origmag.*phase;
newsignal=ifft(newft);
for k=1:z
    for i= 1:iter(k)
        newft=origmag.*exp(1j.*phase);
        newsignal=istft(newft,'Window',windowtype, 'OverlapLength',overlap,'FFTLength', fftlength);
        newsignal=real(newsignal);
        newft=stft(newsignal,'Window',windowtype, 'OverlapLength',overlap,'FFTLength', fftlength);
        phase=angle(newft);
    end
 outputsignal=istft(origmag.*phase,'Window',windowtype, 'OverlapLength',overlap,'FFTLength', fftlength);

outputsignal=abs(outputsignal); %take real part of istft with phase
outputreal=audioplayer(outputsignal, Fs);
%play(outputreal);
filename=strcat('stftoutput_',string(iter(k)),'.wav')
audiowrite(filename, outputsignal,Fs) 
difference=origphase-phase;
diff=[diff norm(difference)]  
end


figure;
stem(iter,diff) %plotting frobenius norm
xlabel('Number of Iterations')
ylabel('Frobenious Norm of Phase Difference')
title('Frobenious Norm of Phase Difference vs Number of Iterations')
xlim([0 600] )

