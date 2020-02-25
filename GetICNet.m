function net = GetICNet(v_fStrain,m_fYtrain)
% Generate trained interference cancellation network
%
% Syntax
% -------------------------------------------------------
% net = GetICNet(v_fStrain,m_fYtrain)
%
% INPUT:
% -------------------------------------------------------
% v_fStrain - training labels
% m_fYtrain - training inputs (channel outputs + interference
%
% OUTPUT:
% -------------------------------------------------------
% net  - trained neural network

% Generate neural network
inputSize = size(m_fYtrain,1);
numHiddenUnits = 60;
numClasses = 2; % Binary constellations

% Nir - work around converting LSTMs into a perceptron with sigmoid activation
LSTMLayer = lstmLayer(numHiddenUnits,'OutputMode','last'...
    ...,'RecurrentWeights', zeros(4*numHiddenUnits,numHiddenUnits)...
    , 'RecurrentWeightsLearnRateFactor', 0 ...
    , 'RecurrentWeightsL2Factor', 0 ...
    );
LSTMLayer.RecurrentWeights = zeros(4*numHiddenUnits,numHiddenUnits);

% Layers = 3 fullly connected + softmax    
layers = [ ...
    sequenceInputLayer(inputSize)    
    LSTMLayer
    fullyConnectedLayer(floor(numHiddenUnits/2))
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% Train network
net = TrainICNet(v_fStrain,m_fYtrain, layers, 0);