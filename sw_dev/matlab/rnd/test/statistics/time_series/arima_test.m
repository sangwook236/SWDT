%-----------------------------------------------------------
% REF [site] >> https://kr.mathworks.com/help/econ/arima-class.html

% Specify an ARIMA(2,1,2) model.
Mdl = arima(2, 1, 2);
% Specify an AR(3) model with known coefficients.
Mdl = arima('Constant', 0.05, 'AR',{0.6, 0.2, -0.1}, 'Variance', 0.01);
% Specify an MA model with no constant, and moving average terms at lags 1, 2, and 12.
Mdl = arima('Constant', 0, 'MALags',[1, 2, 12]);
% Specify a multiplicative seasonal ARIMA model with seasonal and nonseasonal integration.
Mdl = arima('Constant', 0, 'D', 1, 'Seasonality', 12, 'MALags', 1, 'SMALags', 12);
% Specify the ARIMAX(1,1,1) model.
Mdl = arima('AR', 0.2, 'D', 1, 'MA', 0.3, 'Beta', 0.5);
% Specify an ARIMA(1,0,1) conditional mean model with a GARCH(1,1) conditional variance model.
Mdl = arima(1,0,1);  % Specify the conditional mean model.
Mdl.Variance = garch(1,1)  % Specify the conditional variance model.

% Estimate parameters of regression models with ARIMA errors.
[EstMdl, EstParamCov, logL, info] = estimate(Mdl, y)

% Infer innovations of regression models with ARIMA errors.
[E, U, V, logL] = infer(Mdl, Y);

% Forecast responses of regression model with ARIMA errors.
[Y, YMSE, U] = forecast(Mdl, numPeriods);

% Monte Carlo simulation of regression model with ARIMA errors.
[Y, E, U] = simulate(Mdl, numObs);

%-----------------------------------------------------------
% REF [site] >> https://kr.mathworks.com/help/econ/regarima-class.html

% Create regression model with ARIMA errors.
Mdl = regARIMA(2, 1, 3);
Mdl = regARIMA('Intercept', 2, 'AR', {0.2 0.3}, 'MA', {0.1}, 'Variance', 0.5, 'Beta', [1.5 0.2]);
Mdl = regARIMA('Intercept', 1, 'Beta', 6, 'AR', 0.2,...
	'MA', 0.1, 'SAR', {0.5,0.2}, 'SARLags', [4, 8],...
	'SMA', {0.05,0.01}, 'SMALags', [4 8], 'D', 1, 'Seasonality', 4, 'Variance', 1);
Mdl = regARIMA('Intercept', 1, 'Beta', 6, 'AR', 0.2,...
	'MA', 0.1, 'SAR', {0.5,0.2}, 'SMA', {0.05,0.01},...
	'D', 1, 'Seasonality', 4, 'Variance', 1);

%-----------------------------------------------------------
% REF [site] >> https://kr.mathworks.com/help/econ/regarima.arima.html

% Specify the regression model with ARMA(4,1) errors.
Mdl = regARIMA('AR', {0.8, -0.4}, 'MA', 0.3, 'ARLags', [1 4], 'Intercept', 1, 'Beta', 0.5, 'Variance', 1);

rng(1);  % For reproducibility.
T = 20;
X = randn(T, 1);

% Convert a regression model with ARIMA time series errors (regARIMA) to a model of type arima including a regression component (ARIMAX).
[ARIMAX, XNew] = arima(Mdl, 'X', X);
