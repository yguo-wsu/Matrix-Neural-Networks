function C = prod_tensor_mat(T, V)
% this function calculate product of a 3-order tensor T and a matrix V such
% the i-th slice matrice C(:,:,i) of tensor C is the product of i-th slice
% T(:,:,i) of T and the transpose of V

T = permute(T,[2,1,3]);  % make transpose for each slice
C = prod_mat_tensor(V, T);
C = permute(C, [2, 1,3]);
end