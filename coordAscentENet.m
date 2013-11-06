function [w0, w] = coordAscentENet(y, X, lambda, alpha, init, nrIterations)
% check input sizes
assert(size(y, 1)==size(X, 1) && size(y, 1) > 1);
n = size(y, 1);
k = size(X, 2);

% initialize parameters
if ~isempty(init)
    %support warm start
    w0 = init{1};
    w = init{2};
else
    w0 = mean(y);
    w = 0.1*ones(k,1);
end

% assume default tolerance and number of iterations
TOL = 1e-5;
if isempty(nrIterations)
    MAXIT = 100;
else
    MAXIT = nrIterations;
end

% tracking likelihood
lls = zeros(MAXIT,1);
prevll = -realmax;


ll = loglik(y, X, lambda, alpha, w0, w);
iter = 0;

%close all;
%figure(1);
%clf
%ax1 = axes('Position',[0.1 0.85 0.7 0.045])
%ax2 = axes('Position',[0.1 0.8 0.7 0.045]);
%ax3 = axes('Position',[0.81 0.3 0.09 0.49]);
%ax4 = axes('Position',[0.1 0.3 0.7 0.49]);
%ax5 = axes('Position',[0.1 0.05 0.8 0.2]);
while ll - prevll > TOL && iter < MAXIT
    iter = iter+1;
    prevll = ll;
    
    % updates
    w0 = 1/n*sum(y - X*w);
    for j=1:k
        w(j) = 0;
        B = sum(X(:,j).^2);
        if B == 0
            w(j) = 0;
        else
            w(j) = 1/(B + lambda*(alpha)) * shrinkThreshold((y - w0 - X*w)'*X(:,j),(1-alpha)*lambda);
        end
    end
    
    % likelihood for new state
    ll = loglik(y,X,lambda,alpha,w0,w);
    
    
    if ll-prevll >= TOL
        assert(ll-prevll>=0)
    end
    
    lls(iter) = ll;
    
    %axes(ax1);cla;
    %imagema(y');
    %title(['ElasticNet fit with $$\sum_i (y_i - \beta_0 - x_i''\beta)^2 - '...
    %       num2str(alpha*lambda/2) '\sum_j \beta_j^2 - ' ...
    %       num2str((1-alpha)*lambda) '\sum_j |\beta_j|$$'],'Interpreter','Latex');
        
    %set(gca,'XTick',[],'YTick',1,'YTickLabel','y','TickLength',[0 0]);
    
    %axes(ax2);cla;
    %imagema((beta0 + x*beta)');
    %set(gca,'XTick',[],'YTick',1,'YTickLabel','beta0 + X*beta','TickLength',[0 0]);
    
    %axes(ax3);cla;
    %barh(flipud(beta));
    %title('beta')
    %set(gca,'YTick',[]);
    %ylim([0.5 length(beta)+0.5])
    %xlim([-max(abs(beta)+1e-3) max(abs(beta)+1e-3)])
    
    %axes(ax4);cla;
    %imagema(x');
    %set(gca,'XTick',[],'YTick',[size(x,2)/2],'YTickLabel','X');
    
    %axes(ax5);cla;
    %plot(lls(1:iter))
    %xlabel('iteration');
    %ylabel('penalized log-likelihood');
end

function ll = loglik(y, x, lambda, alpha, w0, w)
n = size(y,1);
ll = -1/2*(y - w0 - x*w)'*(y - w0 - x*w)...
    -alpha*lambda/2*sum(w.^2) - (1-alpha)*lambda*sum(abs(w));

function x = shrinkThreshold(x, lambda)
btemp = abs(x);
maxval = max(abs(x) - lambda, 0);
x = sign(x).*max(abs(x) - lambda, 0);

