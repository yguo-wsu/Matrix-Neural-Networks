function [W] = initializeMatrixWeights(arch)
% Weight initialiaztions for different updates and
% for different-layer architectures.

N = length(arch.layerSize);
L = N-1;
layerSize = arch.layerSize;

U = cell(1,L-1);
V = cell(1,L-1);
B = cell(1,L-1);

% Initialize the last soft-max connection coefficient
U_bar = 0.01*randn(layerSize{L}.I, layerSize{L}.J, layerSize{N});
U_bar = reshape(U_bar, layerSize{L}.I * layerSize{L}.J, layerSize{N});
U_bar = U_bar ./ repmat( sqrt( sum(U_bar.^2,1) ), [layerSize{L}.I * layerSize{L}.J ,1] );
U_bar = reshape(U_bar, layerSize{L}.I, layerSize{L}.J, layerSize{N});
b_bar = zeros(layerSize{N},1);

for i=1:L-1
    U{i} = 0.01*randn(arch.pool * layerSize{i+1}.I, layerSize{i}.I);
    U{i} = U{i}./repmat(sqrt(sum(U{i}.^2,2)), [1, layerSize{i}.I]);
    V{i} = 0.01*randn(arch.pool * layerSize{i+1}.J, layerSize{i}.J);
    V{i} = V{i}./repmat(sqrt(sum(V{i}.^2,2)), [1, layerSize{i}.J]);
    B{i} = zeros(arch.pool * layerSize{i+1}.I, arch.pool * layerSize{i+1}.J);
end;

W.U = U;
W.V = V;
W.B = B;
W.U_bar = U_bar;
W.b_bar = b_bar;

end

