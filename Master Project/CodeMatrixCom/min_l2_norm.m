function [ f,g ] = min_l2_grad( C,R,M )

temp = M.*(R-C*C');
f= (norm(temp(:),2))^2;

temp2 = -4.*((M.*(R-C*C'))*C);
g=temp2(:);
end

