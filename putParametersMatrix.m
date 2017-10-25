function [ W] = putParametersMatrix( Wv, layerSize, pool)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

LL = length(layerSize);
L = LL - 1;
U = cell(1,L-1);
V = cell(1,L-1);
B = cell(1,L-1);

count_start = 1;
for i=1:L-1
    count_end = count_start + pool*layerSize{i+1}.I*layerSize{i}.I - 1;
    U{i} = reshape(Wv(count_start:count_end), pool*layerSize{i+1}.I, layerSize{i}.I);
    count_start = count_end + 1;
    count_end = count_start+pool*layerSize{i+1}.J*layerSize{i}.J - 1;
    V{i} = reshape(Wv(count_start:count_end), pool*layerSize{i+1}.J, layerSize{i}.J);
    count_start = count_end + 1;
    count_end = count_start+pool*layerSize{i+1}.I*pool*layerSize{i+1}.J - 1;
    B{i} = reshape(Wv(count_start:count_end), pool*layerSize{i+1}.I, pool*layerSize{i+1}.J);
    count_start = count_end + 1;
end;
count_end = count_start + layerSize{L+1}*layerSize{L}.I*layerSize{L}.J - 1;
U_bar = reshape(Wv(count_start:count_end), layerSize{L}.I, layerSize{L}.J, layerSize{L+1});
count_start = count_end + 1;
count_end = count_start + layerSize{L+1} - 1;
b_bar = Wv(count_start:count_end);

W.U = U;
W.V = V;
W.B = B;
W.U_bar = U_bar;
W.b_bar = b_bar(:);
end

