close all; clearvars; clc;
%% Parameters
par.gam= 2; % CRRA parameter for intertemporal utility
par.rho = 0.05; % discount rate
par.eta = 0.0; % HRRA parameter

% choose country from LIC (low-income country), MIC, HIC
country = "MIC";

par.alph = .3; % capital intensity
par.del = 0.1; % depreciation

% level of development 
% with constant rates 1.8%, to make the TFP 4x: devel + 77, 6x: devel+100, to make it 8x:
% devel + 115; 16x: 151
if country == "LIC"
    devel = -20;
elseif country == "MIC"
    devel = 57;
elseif country == "HIC"
    devel = 80;
end

% from Comin et. el (CLM) paper
g0 = 0.013; %growth rate of TFP in investment goods sector
ga = 0.029; %growth rate of TFP in agricultural goods sector
gm = 0.013; %growth rate of TFP in manufacturing goods sector
gs = 0.011; %growth rate of TFP in services sector

par.g0 = log(1+g0); %growth rate of TFP in investment goods sector
par.ga = log(1+ga); %graddpath ~/Documents/MATLAB/chebfun-master/owth rate of TFP in agricultural goods sector
par.gm = log(1+gm); %growth rate of TFP in manufacturing goods sector
par.gs = log(1+gs); %growth rate of TFP in services sector


% prices matched to shares in log(rgdp per capita) for middle-income
% roughly based on the handbook chapter on structural transformation by Rogerson et. al.
par.A0 = exp(devel*par.g0);
par.Aa = .4*exp(devel*par.ga);
par.Am = .7*exp(devel*par.gm);
par.As = exp(devel*par.gs);

par.sig = .5; % elasticity of substitution between goods
par.epsilon = (1-par.sig)*[.05,1.1,1.2]; % income elasticity of demand
%par.epsilon = (1-par.sig)*[1,1,1]; % homothetic case

par.omega = [1,1,1]/length(par.epsilon);

par.I=500; % size of the wealth grid
par.amin = 0; % minimum wealth 
par.amax = max(10*(devel - 40),10); % upper wealth increases with devel, is 10 for the poorest; 
% doesn't matter much; enough that the saturation wealth level falls within

mobil_gap = 3; % this controls the dispersion in productivity while normalising the average productivity = 1
par.z1 = .5; % low state productivity
par.z2 = mobil_gap + 1 - mobil_gap*par.z1; %high state productivity: to keep the total labour = 1
par.z = [par.z1,par.z2];
par.lambda1 = 1; % transition rate from low state
par.lambda2 = mobil_gap*par.lambda1; % transition rate from high state

cellfun(@(x) assignin('base', x, par.(x)), fieldnames(par));

par.maxit=30;
par.crit = 1e-12; % criteria for convergence of value function
par.Delta = 1000; %time-step for the implicit method

da = (amax-amin)/(I-1);
a = linspace(amin,amax,I)'; % linear grid
zz = ones(I,1)*z;
aa = [a,a];

%% Solving for the steady state
% function which computes the excess demand for capital as a function of the interest rate
exd = @(r) exd_NHCES(par, r); 

r0 = 0.025; % initial guess for the interest rate

tic;
options=optimset('MaxIter', 20, 'Display', 'iter', 'PrecondBandWidth', 1,'TolFun', 1e-5,'TolX',1e-7);
disp('Solving for the stationary equilibrium r, w & p...');
warning('off', 'MATLAB:singularMatrix'); 
[r,fval,exitflag,output] = fsolve(exd, r0, options); % use fsolve to solve for the interest rate which clears market
[exdem, w, p, c, expenditure, s_rate, no_wealth, g, Ks, v, welfare, L, L_alloc, e_inv, h_inv, expend_ch, AT] = exd(r);
duration = toc;

par.e_inv = e_inv; % indirect utility function using prices at the steady state
par.expend_ss = expend_ch; % expenditure function
par.p_ss = p; % goods prices at the steady state
par.r = r; % interest rate at ss
par.w = w; % wage at ss

share_emp_a = L_alloc(2); % agriculture share of employment
share_emp_s = L_alloc(4); % services share of employment

if (abs(fval) < 1e-4)
    fprintf('Converged with accuracy %.1d in %.1f seconds. \n', fval, duration);
else
    disp('Could not converge; please modify the starting point r0.');
end

%% Computing Additional Moments of Interest
% computing the elasticity of intertemporal substitution across the
% distribution
dom_c = [.1,100]; % domain for utility at the ss
dom_e = e_inv.domain; % domain for expenditure at the ss

MU_ch = diff(e_inv); % marginal utility of expenditure
ME_ch = diff(expend_ch); % marginal expenditure of utility
varepsilon_e = @(e) ME_ch(e_inv(e)).*e_inv(e)./e;
vareps_ch = chebfun(varepsilon_e, dom_e);
vareps_diff = diff(vareps_ch);
eta_e = @(e) vareps_diff(e).*e./vareps_ch(e);

% elasticity of intertemporal substitution across the distribution
eis = 1./((gam + varepsilon_e(expenditure) - 1)./varepsilon_e(expenditure) + eta_e(expenditure)); 

% price index and income across the distribution
p_idx = expenditure./c; % price index across the distribution
y = w.*zz + r.*aa;

% sectoral consumption across the distribution
c_a = omega(1).*((p(1)./expenditure).^(-sig)).*(c.^epsilon(1));
c_m = omega(2).*((p(2)./expenditure).^(-sig)).*(c.^epsilon(2));
c_s = omega(3).*((p(3)./expenditure).^(-sig)).*(c.^epsilon(3));

%%%%%% Aggregate Moments
%%%%%%%%%%%%%%%%%%%%%%%%
Expend = g(:,1)'*expenditure(:,1)*da + g(:,2)'*expenditure(:,2)*da; % aggregate consumption expenditure
C = g(:,1)'*c(:,1)*da + g(:,2)'*c(:,2)*da; % aggregate consumption index
C_ad = g(:,1)'*c_a(:,1)*da + g(:,2)'*c_a(:,2)*da; % aggregate demand for consumption of agricultural goods
C_md = g(:,1)'*c_m(:,1)*da + g(:,2)'*c_m(:,2)*da; % aggregate demand for consumption of manufacturing goods
C_sd = g(:,1)'*c_s(:,1)*da + g(:,2)'*c_s(:,2)*da; % aggregate demand for consumption of services

EIS = g(:,1)'*eis(:,1)*da + g(:,2)'*eis(:,2)*da; % average (expenditure weighted EIS)


share_a = p(1)*C_ad/Expend; % expenditure share of agricultural goods
share_m = p(2)*C_md/Expend; % expenditure share of manufacturing goods
share_s = p(3)*C_sd/Expend; % expenditure share of services

kl = Ks/L; %capital-labour ratio
par.kl = kl;

% sectoral output
Y0_pf = A0*((kl)^alph)*L_alloc(1);
Ya_pf = Aa*((kl)^alph)*L_alloc(2);
Ym_pf = Am*((kl)^alph)*L_alloc(3);
Ys_pf = As*((kl)^alph)*L_alloc(4);

Invest = A0*(kl^alph)*L_alloc(1); % investment
GDP_exp = Expend + Invest; % gdp (nominal)
GDP = A0*(kl^alph)*L; % GDP
Invest_rate = Invest/GDP;

P_idx = Expend/C; % aggregate (average price index)

KY = Ks/GDP; % capital-output ratio

rGDP = GDP/P_idx; % real gdp

%% Distributional Moments

% income share of high types:
Yshare_h = (g(:,2)'*y(:,2)*da)/GDP;

% wealth and income ineq
II = 2*I;
gg = reshape(g, II, 1);
g_a_cont = gg(1:I)+gg(I+1:II);
g_a = g_a_cont*da;
cum_g_a = cumsum(g_a.*a)/sum(g_a.*a);
trapez_a = (1/2)*(cum_g_a(1)*g_a(1) + sum((cum_g_a(2:I) + cum_g_a(1:I-1)).*g_a(2:I)));
Gini_w = 1 - 2*trapez_a; % Gini_wealth

%Gini of capital income
yk = r.*a;
g_yk = g_a;

%Gini of total income
yy =reshape(y,II,1);
[yy,index] = sort(yy);
g_y = gg(index)*da;

% consumption(utility) price index
pp_idx =reshape(p_idx,2*I,1);
[pp_idx,index] = sort(pp_idx,'d');
g_pp = gg(index)*da;


S_y = cumsum(g_y.*yy)/sum(g_y.*yy);
trapez_y = (1/2)*(S_y(1)*g_y(1) + sum((S_y(2:II) + S_y(1:II-1)).*g_y(2:II)));
Gini_inc = 1 - 2*trapez_y;

G_y = cumsum(g_y);
G_a = cumsum(g_a);

%Top 10% Income Share
p1 = 0.1;
[~, index] = min(abs((1-G_y) - p1));
top_inc = 1-S_y(index);

%Top 10% Wealth Share
p1 = 0.1;
[obj, index] = min(abs((1-G_a) - p1));
top_wealth = 1-cum_g_a(index);

y_mean = sum(yy.*g_y);
a_mean = sum(sum(aa.*g))*da;


x_max_idx = max(find(cum_g_a <= .999)); % cut x at 0.999 percentile of wealth for clearer plots
x_max = a(x_max_idx);

if isempty(x_max)
    x_max = .9*amax;
end
x_min = amin;

%% Sample Plots
x_axis = aa;
if (country == 'LIC')
    xlab = '';
else
    xlab = 'Wealth, a';
end

%fig_extension = '.pdf';

%filename_fig1 = strcat('ss_eis_',country);

figure(1);
plot(x_axis,g_a,'LineWidth',3)
grid
title(strcat('Wealth Distribution (',country,')'), 'Interpreter','latex', 'FontSize',25)
xlabel(xlab, 'Interpreter','latex', 'FontSize',20)
xlim([x_min x_max])
ylim([0, 0.05])
set(gca,'TickLabelInterpreter','latex', 'FontSize', 22, 'FontWeight', 'bold')

%write figure to file
%fig1 = gca;
%exportgraphics(fig1, strcat(filename_fig1,'_1',fig_extension)) 

figure(2)
plot(x_axis,p(1)*c_a./expenditure, 'LineWidth',3)
set(gca,'FontSize',12)
grid
title(strcat('Share of Agriculture (',country,')'), 'Interpreter','latex', 'FontSize',25)
xlabel(xlab, 'Interpreter','latex', 'FontSize',20)
xlim([x_min x_max])
%ylim([0.15, 0.8])
legend('Low-income HH','High-income HH','Interpreter','latex', 'FontSize',15,Location='east')
set(gca,'TickLabelInterpreter','latex', 'FontSize', 22, 'FontWeight', 'bold')


figure(3);
plot(x_axis,p(3)*c_s./expenditure, 'LineWidth',3)
set(gca,'FontSize',12)
grid
title(strcat('Share of Services (',country,')'), 'Interpreter','latex', 'FontSize',25)
xlabel(xlab, 'Interpreter','latex', 'FontSize',20)
%ylim([0.08, 0.52])
xlim([x_min x_max])
legend('Low-income HH','High-income HH','Interpreter','latex', 'FontSize',15,Location='east')
set(gca,'TickLabelInterpreter','latex', 'FontSize', 22, 'FontWeight', 'bold')


figure(4)
plot(x_axis,eis, 'LineWidth',3)
if (country == 'LIC')
    yline(EIS, '--r', 'Avg EIS','Interpreter','latex', 'FontSize',18);
else
    yline(EIS, '--r', 'Avg EIS','Interpreter','latex', 'FontSize',18, LabelVerticalAlignment='bottom');
end
set(gca,'FontSize',12)
grid
title(strcat('EIS (',country,')'), 'Interpreter','latex', 'FontSize',25)
xlabel(xlab, 'Interpreter','latex', 'FontSize',20)
xlim([x_min x_max])
%ylim([0.15, 0.5])
legend('Low-income HH','High-income HH','Interpreter','latex', 'FontSize',15,Location='east')
set(gca,'TickLabelInterpreter','latex', 'FontSize', 22, 'FontWeight', 'bold')

figure(5);
plot(x_axis,p_idx,'LineWidth',3)
yline(P_idx, '--r', 'Avg Cost of Living Index','Interpreter','latex', 'FontSize',18);
set(gca,'FontSize',12)
grid
title(strcat('Cost of Living Indices (',country,')'), 'Interpreter','latex', 'FontSize',25)
xlabel(xlab, 'Interpreter','latex', 'FontSize',20)
xlim([x_min x_max])
legend('Low-income HH','High-income HH','Interpreter','latex', 'FontSize',15)
set(gca,'TickLabelInterpreter','latex', 'FontSize', 22, 'FontWeight', 'bold')


figure(6);
plot(x_axis,s_rate,'LineWidth',2)
yline(0, '--r', 'Aggregate Saving Rate');
set(gca,'FontSize',12)
grid
title('Saving Rate')
xlabel('Wealth, a')
ylabel('Saving rate')
xlim([x_min x_max])
legend('Low-income','High-income')

