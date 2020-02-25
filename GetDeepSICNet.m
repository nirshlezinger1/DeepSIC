function v_cNet = GetDeepSICNet(m_fStrain,m_fYtrain, s_nIter)
% Generate multi-layer trained interference cancellation network
%
% Syntax
% -------------------------------------------------------
% v_cNet = GetDeepPICNet(v_fStrain,m_fYtrain)
%
% INPUT:
% -------------------------------------------------------
% m_fStrain - training labels
% m_fYtrain - training inputs (channel outputs + interference
% s_nIter - number of IC iteration
%
% OUTPUT:
% -------------------------------------------------------
% v_cNet  - (users x iterations) trained neural networks
  
fSymToProb = @(x)0.5*(x+1);    

s_nK = size(m_fStrain,1);

% Generate network for each user
v_cNet = cell(s_nK,s_nIter);
parfor kk=1:s_nK
    % Get trained network -
    %   label = user k symbols.
    %   Input = channel input + remaining symbols converted to Pr( ) = 1
    v_cNet{kk,1} = GetICNet(m_fStrain(kk,:),[m_fYtrain; fSymToProb(m_fStrain([1:kk-1 kk+1:end],:))]);
end
% Add some retraining using network outputs as soft input
m_fP = 0.5*ones(size(m_fStrain));
for jj=2:s_nIter
    m_fPNext = zeros(size(m_fStrain));
    % Apply network
    for kk=1:s_nK
        % Get soft output to use for retraining
        m_Input = num2cell([m_fYtrain; m_fP([1:kk-1 kk+1:end],:)],1);
        m_fPTemp = predict(v_cNet{kk,jj-1},m_Input);
        m_fPNext(kk,:) = m_fPTemp(:,2)';
    end
    
    m_fP = m_fPNext;
    % Retrain network
    parfor kk=1:s_nK
        % Retrain network isomg soft outputs
        v_cNet{kk,jj} = GetICNet(m_fStrain(kk,:),[m_fYtrain; m_fP([1:kk-1 kk+1:end],:)]);
    end
end