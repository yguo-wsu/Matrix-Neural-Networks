function [pred] = infer_MatrixNetwork(W,x,pool,layerSize)
% Inference for 2 layers with the basic architecture.

%classes = size(annot,2);
LL = length(layerSize);
L = LL - 1;


M{1} = x;

for i = 1:L-1
    H{i} =bsxfun(@plus, prod_mat_tensor_mat(W.U{i}, M{i}, W.V{i}),  W.B{i});
    Ir{i} = H{i} < 0;
    % reLU
    R{i} = H{i};
    R{i}(Ir{i}) = 0;
    
    % Sigmoid activation
    R{i} = 1 ./ ( 1 + exp(- H{i}));
    
    % max pooling
    [M{i+1}, Im{i}] = maxPoolMatrix(R{i}, pool);
    M{i+1} = R{i};
end

H{L} = reshape(W.U_bar, size(W.U_bar,1)*size(W.U_bar,2), size(W.U_bar,3))' * reshape(M{L},[layerSize{L}.I*layerSize{L}.J,size(x,3)]) + repmat(W.b_bar,[1,size(x,3)]);

% X{L+1} is the output of output layer, a matrix of k by size(data,3)
maxH = max(H{L},[], 1);
%max(maxH)
M{L+1} = exp(H{L}-repmat(maxH,[layerSize{L+1}, 1])) + 1e-8;
M{L+1} = M{L+1} ./ repmat(sum(M{L+1}),[layerSize{L+1},1]);

pred = M{L+1};
%M{L+1} = exp(H{L});
%pred = M{L+1} ./ repmat(sum(M{L+1}),[layerSize{L+1},1]);

end