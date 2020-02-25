% Test DeepSIC with binary constellations 
clear all;
close all;
clc;

rng(1);

 
global v_fConst; 
%% Parameters setting
s_nN  = 4;             % Number of Rx antennas
s_nK  = 4;             % Number of transmitted symbols
v_fSNRdB =  0:2:14;    % SNR values in dB.
s_nIter = 5;            % Interference cancellation iterations
s_fTrainSize = 5000;    % Training size
s_fTestSize = 20000;    % Test data size
 

s_fEstErrVar = 0.1;   % Estimation error variance
% Frame size for generating noisy training
s_fFrameSize = 500; 
s_fNumFrames = s_fTrainSize/s_fFrameSize;
s_fNumTestFrames = s_fTestSize/s_fFrameSize;

% Select which decoder to simulate
v_nCurves   = [...          % Curves 
    1 ...                   % Soft IC, perfect CSI  
    1 ...                   % Soft IC, CSI uncertainty
    1 ...                   % Seq. DeepSIC, perfect CSI
    1 ...                   % Seq. DeepSIC, CSI uncertainty
    ];

s_nCurves = length(v_nCurves);

v_stPlots = strvcat(  ... 
    'Iterative SIC, perfect CSI',...
    'Iterative SIC, CSI uncertainty',... 
    'Seq. DeepSIC, perfect CSI', ... 
    'Seq. DeepSIC, CSI uncertainty' ... 
    );

% Generate channel matrix
m_fH = zeros(s_nN, s_nK);
for ii=1:s_nN
    for jj=1:s_nK
        m_fH(ii,jj) = exp(-abs(ii-jj));
    end
end

 
% BPSK constellation
v_fConst = [-1 1];
fSymToProb = @(x)0.5*(x+1);  
fProbToSym = @(x)sign(x-0.5); 
 

%% Simulation loop
m_fBER = zeros(s_nCurves, length(v_fSNRdB));

% Generate training symbols - BPSK
m_fStrain = randsrc(s_nK,s_fTrainSize, v_fConst);

% Generate test symbols - BPSK
m_fStest = randsrc(s_nK,s_fTestSize, v_fConst);

% Training with noisy CSI
m_fRtrain = zeros(s_nN, s_fTrainSize);
for kk=1:s_fNumFrames
    Idxs=((kk-1)*s_fFrameSize + 1):kk*s_fFrameSize;
    m_fRtrain(:,Idxs) =  (m_fH.*(1+  sqrt(s_fEstErrVar)*randn(size(m_fH))))*m_fStrain(:,Idxs);
end


for ii=1:length(v_fSNRdB)
    s_fSigW = 10^(-0.1* v_fSNRdB(ii));
    tic;
    % Generate channel outputs
    % Gaussian channel
    m_fYtrain = m_fH * m_fStrain + sqrt(s_fSigW)*randn(s_nN, s_fTrainSize);
    m_fYtrainErr = m_fRtrain + sqrt(s_fSigW)*randn(s_nN, s_fTrainSize);
    m_fYtest = m_fH * m_fStest + sqrt(s_fSigW)*randn(s_nN, s_fTestSize);
    
     
    % Soft PIC detector
    if (v_nCurves(1) == 1)
        % Initial guess - zeros
        m_fInitial = zeros(size(m_fStest));
        m_fBER(1,ii) = s_fSoftPIC(m_fYtest, m_fStest, m_fInitial, m_fH, (s_fSigW)*eye(s_nN), s_nIter);
    end
    
    % Soft PIC detector, uncertainty
    if (v_nCurves(2) == 1)
        for jj=1:s_fNumTestFrames
            v_fIdxs = ((jj-1)*s_fFrameSize+1) : (jj*s_fFrameSize);
            m_fBER(2,ii) = m_fBER(2,ii) + s_fSoftPIC(m_fYtest(:,v_fIdxs), m_fStest(:,v_fIdxs), ...
                                                    zeros(size(m_fStest(:,v_fIdxs))), ...
                                                    (m_fH.*(1+  sqrt(s_fEstErrVar)*randn(size(m_fH)))),...
                                                    (s_fSigW)*eye(s_nN), s_nIter);             
        end
        % Average over noisy channel tests
        m_fBER(2,ii) = m_fBER(2,ii)/s_fNumTestFrames;        
    end
     
    
    % DeepSIC detector, sequential training 
    if (v_nCurves(3) == 1)        
        % Get network
        v_cNet = GetDeepSICNet(m_fStrain,m_fYtrain, s_nIter);
        
        % Apply network
        m_fBER(3,ii) = s_fDetDeepSIC(m_fYtest, m_fStest, v_cNet, s_nIter);
    end
     
    
    % DeepSIC detector, sequential training, uncertainty
    if (v_nCurves(4) == 1)
        % Get network
        v_cNet = GetDeepSICNet(m_fStrain, m_fYtrainErr, s_nIter);
        
        % Apply network
        m_fBER(4,ii) = s_fDetDeepSIC(m_fYtest, m_fStest, v_cNet, s_nIter);
    end
    
  
    
    toc; 
    ii
end

%% Display results
 v_stPlotType = strvcat( '-rs', '--ro', '-b^',  '--bv', '-k<', '-g<',...
    '--k>','--g>', '-m*', '--mx',  '-c^', '--cv');

v_stLegend = [];
fig1 = figure;
set(fig1, 'WindowStyle', 'docked');
%
for aa=1:s_nCurves
    if (v_nCurves(aa) ~= 0)
        v_stLegend = strvcat(v_stLegend,  v_stPlots(aa,:));
        semilogy(v_fSNRdB, m_fBER(aa,:), v_stPlotType(aa,:),'LineWidth',1,'MarkerSize',10);
        hold on;
    end
end

xlabel('SNR [dB]');
ylabel('BER');
grid on;
legend(v_stLegend,'Location','SouthWest');
hold off;

