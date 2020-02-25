function s_fErr = s_fSoftPIC(m_fY, m_fS, m_fInitial, m_fH, m_fCovW, s_nIter)
% Soft parallel interference cancellation detector (equiprobable BPSK constellation)
% Inspired by: Cohi, Cheong, and Cioffi, "Iterative soft interference cancellation for multiple antenna systems"
%
% Syntax
% -------------------------------------------------------
% s_fErr = s_fSoftPIC(m_fY, m_fS, m_fInitial, m_fH, m_fCovW, s_nIter)
%
% INPUT:
% -------------------------------------------------------
% m_fY - channel outputs
% m_fS - channel inputs
% m_fH - channel matrix
% m_fInitial - initial guess
% m_fCovW - noise covariance
% s_nIter - number of iterations
%
% OUTPUT:
% -------------------------------------------------------
% s_fErr  - average bit error rate
global v_fConst;

[s_nK, s_nSymbols] = size(m_fS);


fProbToSym = @(x)sign(x-0.5);

% Initialize decisions to zeros
m_fShat = zeros(size(m_fS));

v_fPYgS = zeros(2,1);

% Loop over symbols
for ll=1:s_nSymbols
    % Initialize probability of symbol 1
    v_fP = (m_fInitial(:,ll) +1)/2;
    % Loop over Soft IC iterations
    for ii=1:s_nIter
        % Compute expected values and variances from posteriors
        v_fX = 2*v_fP - 1;
        v_fE = 1 - v_fX.^2;
        % Compute equivalent covariance matrix and postulated interference canceled output
        v_fYk = m_fY(:,ll) - m_fH*v_fX;
        m_fCovk = m_fCovW;
        for jj=1:s_nK
            m_fCovk = m_fCovk + v_fE(jj)* m_fH(:,jj)* m_fH(:,jj)';
        end
        % Loop over users
        for kk=1:s_nK
            % update relevant equivalent covariance matrix and postulated interference canceled output
            v_fYcur = v_fYk + m_fH(:,kk)*v_fX(kk);
            m_fCovcur = m_fCovk - v_fE(kk)* m_fH(:,kk)* m_fH(:,kk)';
            % Compute likelihhod
            v_fPYgS(1) = mvnpdf(v_fYcur',  v_fConst(2)*m_fH(:,kk)', m_fCovcur);
            v_fPYgS(2) = mvnpdf(v_fYcur',  v_fConst(1)*m_fH(:,kk)', m_fCovcur);
            
            % Convert likelihhod to posterior probability
            % Not get NAN
            if (sum(v_fPYgS) > 1e-100)
                v_fP(kk) =  v_fPYgS(1) / sum(v_fPYgS);
            end
            
        end
    end
    % Hard decision
    m_fShat(:,ll) = fProbToSym(v_fP);
end

% Compute BER
s_fErr = mean(mean(m_fShat ~= m_fS));
