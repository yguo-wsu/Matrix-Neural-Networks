function [ Wv ] = getParametersMatrix( W, layerSize)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

LL = length(layerSize);
L = LL - 1;

Wv = [];
for i=1:L-1
    Wv = [Wv, W.U{i}(:)', W.V{i}(:)', W.B{i}(:)'];
end;
Wv = [Wv, W.U_bar(:)' W.b_bar(:)'];
Wv = Wv';
end

