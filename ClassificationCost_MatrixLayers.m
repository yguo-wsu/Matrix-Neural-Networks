function [cost,grads_v] = ClassificationCost_MatrixLayers(Wv,x,annot,pool,layerSize)
% Forward pass for 1 layer with the basic architecture.

LL = length(layerSize);
L = LL - 1;
numData = size(x,3);
%classes = size(annot,2);
H = cell(1,L);
R = cell(1,L-1);
M = cell(1,L+1);
Im = cell(1,L-1);
Ir = cell(1,L-1);
gradU = cell(1,L-1);
gradV = cell(1,L-1);
gradB = cell(1,L-1);
W = putParametersMatrix(Wv, layerSize, pool);
M{1} = x;

for i = 1:L-1
    H{i} =bsxfun(@plus, prod_mat_tensor_mat(W.U{i}, M{i}, W.V{i}),  W.B{i});
    % reLU:   This does not work well. Sometimes get NAN cost.
    Ir{i} = H{i} < 0;
    R{i} = H{i};
    R{i}(Ir{i}) = 0;
    
    % Sigmoid activation
    R{i} = 1 ./ ( 1 + exp(- H{i}));
    
    %R{i} = H{i};
    %  max pooling
    %[M{i+1}, Im{i}] = maxPoolMatrix(R{i}, pool);
    M{i+1} = R{i};
end

%% classifier softmax

H{L} = reshape(W.U_bar, size(W.U_bar,1)*size(W.U_bar,2), size(W.U_bar,3))' * reshape(M{L},[layerSize{L}.I*layerSize{L}.J,numData]) + repmat(W.b_bar,[1,numData]);

% X{L+1} is the output of output layer, a matrix of k by size(data,3)
maxH = max(H{L},[], 1);

M{L+1} = exp(H{L}-repmat(maxH,[layerSize{L+1}, 1])) + 1e-8;
%M{L+1} = exp(H{L});
M{L+1} = M{L+1} ./ repmat(sum(M{L+1}),[layerSize{L+1},1]);



%gradient computation
loss = -annot'.* log(M{L+1});
cost = sum(loss(:)) / numData ;
if isnan(cost)
    save class_p1 W H M
end

outderv = -1 *(annot'-M{L+1}) / numData;   %loss delta or output delta

% Ubar and Bbar gradient
ip_delta = reshape(reshape(W.U_bar, size(W.U_bar,1)*size(W.U_bar,2), size(W.U_bar,3)) * outderv, [layerSize{L}.I, layerSize{L}.J, numData]);  % theta' * outderv;
%thetagrad = outderv * m2';
gradUbar = reshape(M{L},[layerSize{L}.I*layerSize{L}.J,numData])*outderv';   %/numData ;
gradb_bar = sum(outderv,2);

grads.U_bar = gradUbar;
grads.b_bar = gradb_bar(:);

for i = L-1:-1:1
    % upsample through max pool
    %         [rowR, colR, lenR] = size(R{i});
    %         [rowIp, colIp, lenIp] = size(ip_delta);
    %         delta1 = zeros(rowR, colIp, lenIp);
    %         for ii = 1:pool
    %            delta1(ii:pool:rowR, :, :) = ip_delta;
    %         end
    %         delta = zeros(rowR, colR, lenR);
    %         for ii = 1:pool
    %             delta(:,ii:pool:colR,:) = delta1;
    %         end
    %         delta = delta .* Im{i};
    
    %through reLU
    % delta = delta .* double(~(Ir{i}));
    
    %through sigmoid
    delta = ip_delta;
    delta = delta .* R{i} .* (1 - R{i});
    
    gradU{i} = sum_prod_tensors(prod_tensor_mat(delta, W.V{i}'), permute(M{i}, [2,1,3]) );  %/numData ;
    gradV{i} = sum_prod_tensors(prod_tensor_mat(permute(delta,[2,1,3]), W.U{i}'), M{i} );  %/numData ;
    gradB{i} = sum(delta,3);
    ip_delta = prod_mat_tensor_mat(W.U{i}', delta, W.V{i}');
end
grads.U = gradU;
grads.V = gradV;
grads.B = gradB;
grads_v =  getParametersMatrix(grads, layerSize);
end