function [Woptim,infos] = trainMatrixNetwork(W,train_x,train_y,test_x,test_y,val_x,val_y,pool,lambda,numBatches,mbatch,shuffle,numLayers,method,type)

if strcmp(type,'Arch1')
    % Basic architecture
    [Woptim,infos] = trainMatrixNetworkBasic(W,train_x,train_y,test_x,test_y,val_x,val_y,pool,lambda,numBatches,mbatch,shuffle,numLayers,method);
elseif strcmp(type,'Arch2')
    % Basic architecture + batch normalization
    [Woptim,infos] = trainNetworkwithBN(W,train_x,train_y,test_x,test_y,val_x,val_y,pool,lambda,numBatches,mbatch,shuffle,numLayers,method);
end

end

