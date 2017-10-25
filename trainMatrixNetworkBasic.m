function [Woptim,infos] = trainMatrixNetworkBasic(W,train_x,train_y,test_x,test_y,val_x,val_y,pool,lambda,numBatches,mbatch,shuffle,layerSize,method)
traj = zeros(numBatches,1);
LL = length(layerSize);
L = LL - 1;
Wv = getParametersMatrix(W, layerSize);

% Random data shuffling
if shuffle
    indData = randperm(size(train_x,3),size(train_x,3)); % Random shuffling
else
    indData = 1 : size(train_x,1);
end

for b = 1:numBatches % Run through minibatches
    ind = indData((b-1)*mbatch+1:b*mbatch);
    x = double(train_x(:,:,ind));
    annot = double(train_y(ind,:));
    
    [cost, gradsW_v] = ClassificationCost_MatrixLayers(Wv,x,annot,pool,layerSize);
    
    % Riemannian gradient
    if strcmp(method,'UN')
        W1Rgrad = (problemW1.M.egrad2rgrad(W1', W1grad'))';
        %weight update
        W1 = (problemW1.M.retr(W1', -lambda*W1Rgrad'))';
        theta = theta - lambda * thetagrad;
    elseif strcmp(method,'SM')
        W1 = W1 - lambda * sparse(diag(diag(W1*W1'))) * W1grad;
        theta = theta - lambda * thetagrad * sparse(diag(diag(theta'*theta)));
    elseif strcmp(method,'B-SGD')
        Wv = Wv -  lambda * gradsW_v;
    end
    
    
    traj(b) = cost;
    
end
Woptim = putParametersMatrix(Wv, layerSize, pool);

%% Train Test Error Computation
[train_err,test_err,val_err] = computeMatrixTrainTestErrorBasic(W,train_x,train_y,test_x,test_y,val_x,val_y,pool,mbatch,layerSize);
infos.optimcost = mean(traj);
infos.trainerror = train_err;
infos.testerror = test_err;
infos.validerror = val_err;
end