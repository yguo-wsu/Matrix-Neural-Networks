function C = prod_mat_tensor(U, T)
% this function calculate product of a matrix U and a 3-order tensor T such
% the i-th slice matrice C(:,:,i) of tensor C is the product of U and i-th
% slice T(:,:,i) of T,  C(:,:,i) = U*T(:,:,i)

[n1,n2,n3] = size(T);
[m1, m2] = size(U);
if m2~=n1
    error('dimension does not match in product of matrix and tensor!')
end
T = reshape(T, n1, n2*n3);
C = U * T;
C = reshape(C,m1, n2, n3);
end