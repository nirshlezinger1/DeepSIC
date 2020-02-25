function net = TrainICNet(v_fStrain, m_fYtrain, layers, learnRate)
% Train interference cancellation network
%
% Syntax
% -------------------------------------------------------
% net = TrainICNet(v_fStrain, m_fYtrain, layers, learnRate)
%
% INPUT:
% -------------------------------------------------------
% v_fStrain - training labels
% m_fYtrain - training inputs 
% layers - network layers
% learnRate - learning rate (0 for default)
%
% OUTPUT:
% -------------------------------------------------------
% net  - trained neural network

% Set each channel input as a single unique category
v_fScat = categorical(v_fStrain');
m_fYcat = num2cell(m_fYtrain,1)';

if (learnRate == 0)
    learnRate = 0.01;
end
% Train netowrk
maxEpochs = 100;
miniBatchSize = 27;

options = trainingOptions('adam', ... 
    'ExecutionEnvironment','cpu', ...
    'InitialLearnRate', learnRate, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false ...
     );%,'Plots','training-progress'); %);%

net = trainNetwork(m_fYcat,v_fScat,layers,options);