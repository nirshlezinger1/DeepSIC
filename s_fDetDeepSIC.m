function s_fErr = s_fDetDeepSIC(m_fY, m_fS, v_cNet, s_nIter)
% Deep soft parallel interference cancellation detector (equiprobable BPSK constellation) 
%
% Syntax
% -------------------------------------------------------
% s_fErr = s_fDeepPIC(m_fY, m_fS, v_cNet, s_nIter)
%
% INPUT:
% -------------------------------------------------------
% m_fY - channel outputs
% m_fS - channel inputs
% v_cNet - neural networks matrix
% s_nIter - number of iterations
%
% OUTPUT:
% -------------------------------------------------------
% s_fErr  - average bit error rate
 
fProbToSym = @(x)sign(x-0.5);     

[s_nK, s_nSymbols] = size(m_fS);

% Initialize decisions to zeros
m_fShat = zeros(size(m_fS)); 

% Loop over symbols
for ll=1:s_nSymbols
    % Initialize probability of symbol 1
    v_fP = 0.5*ones(s_nK, 1);
    v_fPnext = zeros(size(v_fP));
    v_fCurY = m_fY(:,ll);
    % Loop over Soft IC iterations
    for ii=1:s_nIter
        % Loop over users
        for kk=1:s_nK
            % Apply network
            v_Input = num2cell([v_fCurY; v_fP([1:kk-1 kk+1:end])],1); 
            v_fPTemp = predict(v_cNet{kk,ii},v_Input);             
            % Save on Pr( ) = 1
            v_fPnext(kk) = v_fPTemp(2);
        end
        % Save predictions
        v_fP = v_fPnext;        
    end
    % Hard decision
    m_fShat(:,ll) = fProbToSym(v_fP);
end

% Compute BER
s_fErr = mean(mean(m_fShat ~= m_fS));
