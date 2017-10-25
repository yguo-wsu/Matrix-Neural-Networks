function [train_err,test_err,val_err] = computeMatrixTrainTestErrorBasic(W,train_x,train_y,test_x,test_y,val_x,val_y,pool,mbatch,layerSize)

numBatches = size(train_x,3)/mbatch;%needs to be an integer!
train_err = zeros(numBatches,1);
for b = 1:numBatches %run through minibatches
    ind = (b-1)*mbatch+1:b*mbatch;%deterministic
    
    %train error
    x = double(train_x(:,:,ind));
    annot = double(train_y(ind,:));
    
    %Cost computation
    [pred] = infer_MatrixNetwork(W,x,pool,layerSize);
    
    dd = pred(:) < 0;
    if sum(dd(:)) > 0
        display('somethings wrong here!');
    end
    [~,I] = max(annot,[],2);
    [~,Ip] = max(pred,[],1);
    err = I' ~= Ip;
    train_err(b) = sum(err(:));
end
train_err = sum(train_err(:))/(numBatches*mbatch);%output average error

numBatches = size(test_x,3)/mbatch;%needs to be an integer!
test_err = zeros(numBatches,1);
for b = 1:numBatches %run through minibatches
    ind = (b-1)*mbatch+1:b*mbatch;%deterministic
    
    % test error
    x = double(test_x(:,:,ind));
    annot = double(test_y(ind,:));
    
    %Cost computation
    [pred] = infer_MatrixNetwork(W,x,pool,layerSize);
    
    [~,I] = max(annot,[],2);
    [~,Ip] = max(pred,[],1);
    err = I' ~= Ip;
    test_err(b) = sum(err(:));
end
test_err = sum(test_err(:))/(numBatches*mbatch);%output average error

if ~isempty(val_x)
    numBatches = size(val_x,3)/mbatch;%needs to be an integer!
    val_err = zeros(numBatches,1);
    for b = 1:numBatches %run through minibatches
        ind = (b-1)*mbatch+1:b*mbatch;%deterministic
        
        % test error
        x = double(val_x(:,:,ind));
        annot = double(val_y(ind,:));
        
        %Cost computation
        [pred] = infer_MatrixNetwork(W,x,pool,layerSize);
        [~,I] = max(annot,[],2);
        [~,Ip] = max(pred,[],1);
        err = I' ~= Ip;
        val_err(b) = sum(err(:));
    end
    val_err = sum(val_err(:))/(numBatches*mbatch);%output average error
else
    val_err = [];
end

end