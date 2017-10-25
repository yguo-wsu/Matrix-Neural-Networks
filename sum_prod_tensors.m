function C = sum_prod_tensors(T1, T2)
% this function calculate sum of the product of corresponding slice of two
% 3-order tensors T1 and T2 such
%  C = sum_i ( T1(:,:,1) * T2(:,:,i) )

[n1, n2, n3] = size(T1);
[m1, m2, m3] = size(T2);

T1 = reshape(T1, n1, n2*n3);

T2 = permute(T2,[2,1,3]);  % make transpose for each slice
T2 = reshape(T2, m2, m1*m3);

C = T1 * T2';

end