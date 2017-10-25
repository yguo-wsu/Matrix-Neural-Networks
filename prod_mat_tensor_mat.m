function C = prod_tensor_mat(U, T, V)
% this function calculate product of C = U T V^t such that
% the i-th slice matrice C(:,:,i) of tensor C is the product of U * T(:,:,i) * V^t

C = prod_tensor_mat(prod_mat_tensor(U, T), V);

end