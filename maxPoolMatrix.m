function [m1,Im1] = maxPoolMatrix(r, pool)
% Max pooling operation.
%
dimI = size(r,1);
dimJ = size(r,2);
mbatch = size(r,3);
m1 = zeros(dimI/pool, dimJ/pool, mbatch);
Im1 = zeros(size(r));
tI = zeros(pool*pool,mbatch);
i = 0;
for rows = 1:pool:dimI
    j = 0;
    i = i + 1;
    for cols = 1:pool:dimJ
        %tmp = reshape(r, dimI*dimJ, mbatch);
        j = j + 1;
        if pool == 1
            m = reshape(r(rows:rows+pool-1, cols:cols+pool-1, :), pool*pool, mbatch);
            I = 1:mbatch;
        else
            [m, I] = max(reshape(r(rows:rows+pool-1, cols:cols+pool-1, :), pool*pool, mbatch));
        end
        m1(i,j,:) = m;
        %[II, JJ] = ind2sub([pool, pool], I);
        if pool == 1
            tI(:) = 1;
        else
            tI(:) = 0;
            tI(sub2ind([pool*pool mbatch],I,1:mbatch)) = 1;
        end
        tI = reshape(tI, pool, pool, mbatch);
        Im1(rows:rows+pool-1,cols:cols+pool-1,:) = tI;
        %m1(floor(ll/2)+1,:) = m;
    end
end

end