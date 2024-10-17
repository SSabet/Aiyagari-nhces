% excess demand for capital in 3-sector Aiyagari economy with non-homothetic CES
% preferences (a la Comin et. al. 2021)

function [exdem, w, p, c, expenditure, s_rate, no_wealth, g, Ks, v, welfare, L, L_alloc, e_inv, h_inv, expend_ch, AT] = exd_NHCES(par,r)
%% Parameters
cellfun(@(x) assignin('caller', x, par.(x)), fieldnames(par)); %unpack

a = linspace(amin,amax,I)';
da = (amax-amin)/(I-1);

aa = [a,a];
zz = ones(I,1)*z;

% preallocation
dVF = zeros(I,2);
dVB = zeros(I,2);
c = zeros(I,2);

%% Computing prices, agg capital demand, and labour supply from firms' FOCs and taking r as given
w = (1-alph)*(A0.^(1/(1-alph))).*((alph./(r + del)).^(alph/(1-alph)));
p = A0./[Aa,Am,As]; par.p = p;
kl = (alph/(1-alph))*(w/(r + del));

L = (z1*lambda2 + z2*lambda1)/(lambda1 + lambda2); % supply of efficient labour: inelastic
Kd = L*kl; %demand for capital


%% Function definitions: using Chebyshev approximation for code efficiency
expend = @(c) expend_fun(par,c);  

% with chebfun
dom = [max(0,eta)+1e-2, 100]; % domain to pass to chebfun

expend_ch = chebfun(expend,dom);
e_inv = inv(expend_ch);

u = @(c) (c-eta).^(1-gam)./(1-gam);
d_u = @(c) (c-eta).^gam; % this is inverse of marginal utility

d_e = @(c) de_fun(par,c);
h_foc = @(c) d_e(c).*d_u(c);
h_fun = chebfun(h_foc ,dom);

% this function inverts the derivative of value function to find optimal
% utility
h_inv = inv(h_fun,'splitting','on'); 

%% Initial values
e0 = w*zz + r*aa;

c0 = e_inv(e0);
v0 = u(c0)/rho;
%% Solving for HJB
Aswitch = [-speye(I)*lambda1(1),speye(I)*lambda1;speye(I)*lambda2,-speye(I)*lambda2];

v = v0;
for n=1:maxit
    V = v;
    % forward difference
    dVF(1:I-1,:) = (V(2:I,:)-V(1:I-1,:))/da;
    dVF(I,:) = (c0(I,:)-eta).^(-gam)./d_e(c0(I,:)); %state constraint a<=amax
    % backward difference
    dVB(2:I,:) = (V(2:I,:)-V(1:I-1,:))/da;
    dVB(1,:) = (c0(1,:)-eta).^(-gam)./d_e(c0(1,:)); %state constraint boundary condition
  
    %consumption and savings with backward difference
    cf = h_inv(1./dVF); % use dV to update optimal policy
    ssf = w.*zz + r.*aa - expend(cf);

    %consumption and savings with backward difference
    cb = h_inv(1./dVB);
    ssb = w.*zz + r.*aa - expend(cb);
    
    % dV_upwind makes a choice of forward or backward differences based on
    % the sign of the drift    
    %negative drift --> backward difference
    Ib = ssb <0;
    If = ssf > 0;
    I0 = (1-If-Ib); %at steady state
       
    c = cf.*If + cb.*Ib + c0.*I0;

    util = u(c);
    
    %CONSTRUCT MATRIX
    X = -min(ssb,0)/da;
    Y = -max(ssf,0)/da + min(ssb,0)/da;
    Z = max(ssf,0)/da;
    
    A1=spdiags(Y(:,1),0,I,I)+spdiags(X(2:I,1),-1,I,I)+spdiags([0;Z(1:I-1,1)],1,I,I);
    A2=spdiags(Y(:,2),0,I,I)+spdiags(X(2:I,2),-1,I,I)+spdiags([0;Z(1:I-1,2)],1,I,I);
    A = [A1,sparse(I,I);sparse(I,I),A2] + Aswitch;
    
    B = (1/Delta + rho)*speye(2*I) - A;
    
    u_stacked = [util(:,1);util(:,2)];
    V_stacked = [V(:,1);V(:,2)];
    
    b = u_stacked + V_stacked/Delta;
    V_stacked = B\b; 
    
    V = [V_stacked(1:I),V_stacked(I+1:2*I)];
    
    Vchange = V - v;
    v = V;
    
    dist(n) = max(max(abs(Vchange)));
    if dist(n)<crit
        break
    end
end

expenditure = expend(c);


ss = w.*zz + r.*aa - expenditure;
s_rate = ss./ (w.*zz +r.*aa);

c_a = omega(1).*((p(1)./expenditure).^(-sig)).*(c.^epsilon(1));
c_m = omega(2).*((p(2)./expenditure).^(-sig)).*(c.^epsilon(2));
c_s = omega(3).*((p(3)./expenditure).^(-sig)).*(c.^epsilon(3));


%% Solving for the stationary distribution: KFE
AT = A';
b = zeros(2*I,1);

% Fix one value, o/w the matrix will be singular
b(1) = 1; AT(1,:) = [1,zeros(1,2*I-1)];

%Solve linear system
gg = AT\b;
g_sum = gg'*ones(2*I,1)*da;
gg = gg./g_sum;

g = [gg(1:I),gg(I+1:2*I)];
%% Other Moments
Ks = g(:,1)'*a*da + g(:,2)'*a*da; % aggregate supply of capital

C_ad = g(:,1)'*c_a(:,1)*da + g(:,2)'*c_a(:,2)*da; % aggregate demand for consumption of agricultural goods
C_md = g(:,1)'*c_m(:,1)*da + g(:,2)'*c_m(:,2)*da; % aggregate demand for consumption of manufacturing goods
C_sd = g(:,1)'*c_s(:,1)*da + g(:,2)'*c_s(:,2)*da; % aggregate demand for consumption of services

% welfare
value = g(:,1)'*v(:,1)*da + g(:,2)'*v(:,2)*da; % utilitarian welfare
welfare = -1000/value; % an ad-hoc welfare index, transforming value


% wealth and income distribution
gg = reshape(g, 2*I, 1);
g_a_cont = gg(1:I)+gg(I+1:2*I);
g_a = g_a_cont*da;

G_a = cumsum(g_a); % cumulative distribution of wealth

no_wealth = (G_a(1)+G_a(2))*100; %percentage having zero wealth

output = A0*(kl^alph)*L;
ls_a = (p(1)*C_ad/output); %share of labour force in the agricultural sector
ls_m = (p(2)*C_md/output); %share of labour force in the manufacturing sector
ls_s = (p(3)*C_sd/output); %share of labour force in the secondary sector

La = ls_a * L;
Lm = ls_m * L;
Ls = ls_s * L;
L0 = L-(La + Lm + Ls);

L_alloc = [L0, La,Lm, Ls];

exdem = Kd - Ks;
end

%% Functions
% expenditure as a function of utility, taking prices as given
function ex = expend_fun(par,c)
omega = par.omega;
epsilon = par.epsilon;
p = par.p;
sig = par.sig;

ex = (omega(1).*(c.^epsilon(1)).*(p(1).^(1-sig)));
for i=2:length(p)
    ex = ex + (omega(i).*(c.^epsilon(i)).*(p(i).^(1-sig)));
end
ex = ex.^(1/(1-sig));
end


% marginal expenditure of utility
function de = de_fun(par,c)
omega = par.omega;
epsilon = par.epsilon;
p = par.p;
sig = par.sig;

dims = size(c);
if (dims(1) < Inf)
    c_stacked = reshape(c, prod(dims),1);
else
    c_stacked = c;
end

de = (expend_fun(par,c_stacked).^sig).*(sum(omega.*epsilon.*(c_stacked.^(epsilon-1)).*...
    (p.^(1-sig)),2))/(1-sig);
de = reshape(de,size(c));
end

            
