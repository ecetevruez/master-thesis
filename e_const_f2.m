ts = cputime;

% Input file name
inputFileName = "20_10_1_w.mod";

% Read the contents of the input file
fid = fopen(inputFileName, 'r');
fileData = textscan(fid, '%s', 'Delimiter', '\n'); 
% fclose(fid);
fileData = fileData{1};

% Initialize variables
HorizonLen = 0;
NumThermal = 0;
NumHydro = 0;
NumCascade = 0;
MinSystemCapacity = 0;
MaxThermalCapacity = 0;
MaxSystemCapacity = 0;
Loads = [];
thermalUnitData = [];
rampUp = [];
rampDown = [];
parsingThermalSection = false;
sectionFound = false;
hydroData = [];
waterFlow = {};

% Loop over each line to extract the data
for i = 1:length(fileData)
    line = strtrim(fileData{i});                                             % to remove extra spaces or tabs

    if contains(line, 'HorizonLen')                                          
        HorizonLen = str2double(regexp(line, '\d+', 'match'));               % convert to int
    elseif contains(line, 'NumThermal')
        NumThermal = str2double(regexp(line, '\d+', 'match'));               % convert to int
    elseif contains(line, 'NumHydro')
        NumHydro = str2double(regexp(line, '\d+', 'match')); 
    elseif contains(line, 'NumCascade')
        NumCascade = str2double(regexp(line, '\d+', 'match')); 
    elseif contains(line, 'MinSystemCapacity')
        MinSystemCapacity = str2double(regexp(line, '[\d.]+', 'match'));     % convert to decimal
    elseif contains(line, 'MaxSystemCapacity')
        MaxSystemCapacity = str2double(regexp(line, '[\d.]+', 'match'));     % convert to decimal
    elseif contains(line, 'MaxThermalCapacity')
        MaxThermalCapacity = str2double(regexp(line, '[\d.]+', 'match'));     % convert to decimal
    elseif contains(line, 'Loads')
        loadsData = textscan(fileData{i+1}, '%f');
        Loads = loadsData{1};
    end

    % Check for the "RampConstraints" lines, save the two as rampUp and rampDown
    if startsWith(line, 'RampConstraints')
        % Extract and append the numbers after "RampConstraints"
        tokens = textscan(line, 'RampConstraints %f %f');
        rampUp = [rampUp; tokens{1}];
        rampDown = [rampDown; tokens{2}];
    end
    

    % Check if we are inside the ThermalSection
    if startsWith(line, 'ThermalSection')
        parsingThermalSection = true;
    elseif parsingThermalSection && strcmp(line, 'HydroSection')             % exit termal section if the current line is HydroSection
        break;
    end

    % Extract data inside ThermalSection and ignore rows with RampConstraints
    if parsingThermalSection && ~startsWith(line, 'RampConstraints')
        tokens = str2double(strsplit(line));
        if ~isempty(tokens) && ~any(isnan(tokens))                           % only non empty and not NaN values
            thermalUnitData = [thermalUnitData; tokens];
        end
    end
end

fclose(fid);
fid = fopen(inputFileName, 'r');

while ~feof(fid)
    line = fgetl(fid);
    
    % Check if the HydroSection starts
    if ~isempty(strfind(line, 'HydroSection'))
        sectionFound = true;
        hydroData = [];
        waterFlow = {};
        continue; 
    elseif ~isempty(strfind(line, 'HydroCascadeSection'))
        sectionFound = false;
        continue; 
    end
    
    % If we are inside the HydroSection, read the data
    if sectionFound
        % Split the line into numerical values
        values = str2double(strsplit(line));
        
        % Check if there are valid numerical values
        if ~any(isnan(values))
            hydroData = [hydroData; values];
        else
            % Save every other row as waterFlow
            if ~isempty(values)
                waterFlow{end+1} = line;
            end
        end
    end
end

fclose(fid);

%waterFlow = char(waterFlow); % Convert waterFlow cell array to a character array


% Extract specific columns for hydro unit data and name according to the
% format file
hydroUnitIndex = hydroData(:, 1);        % units
volumeToPower = hydroData(:, 2);         % Liner conversion coefficient from water to power
b_h = hydroData(:, 3);                   % Unused
maxUsage = hydroData(:, 4);              % Max amount of water usable for power generation
maxSpillage = hydroData(:, 5);           % Max amount of spillable water
initialFlood = hydroData(:, 6);          % Initial amount of water into the basin
minFlood = hydroData(:, 7);              % Min amount of water into the basin
maxFlood = hydroData(:, 8);              % Max amount of water into the basin 


% Extract specific columns for thermal unit data and name according to the
% format file
thermalUnitIndex = thermalUnitData(:, 1);        % units
quadraticCoefficients = thermalUnitData(:, 2);   % Quadgencost (power generation cost f^g)
linearCoefficients = thermalUnitData(:, 3);      % Lingencost (power generation cost f^g)
constantCoefficients = thermalUnitData(:, 4);    % Constgencost (power generation cost f^g)
minPower = thermalUnitData(:, 5);                % Plow
maxPower = thermalUnitData(:, 6);                % Pupp
initialStatus = thermalUnitData(:, 7);           % for how long is the unit on/off
minUpTime = thermalUnitData(:, 8);               % minimum up time - L  
minDownTime = thermalUnitData(:, 9);             % minimum down time - l 
coolAndFuelCost = thermalUnitData(:, 10);        % parameter to calculate start up cost f^y
hotAndFuelCost = thermalUnitData(:, 11);         % parameter to calculate start up cost f^y
tau = thermalUnitData(:, 12);                    % parameter to calculate start up cost f^y
tauMax = thermalUnitData(:, 13);                 % parameter to calculate start up cost f^y
fixedCost = thermalUnitData(:, 14);              % parameter to calculate start up cost f^y
SUCC = thermalUnitData(:, 15);                   % unused param
P0 = thermalUnitData(:, 16);                     % s the power that the unit is producing if active prior to the start of the time horizon


% Calculation of the start-up cost
% Cool start-up cost - size 10 1 
coolStartUpCost = coolAndFuelCost .* (1 - exp(-minDownTime ./ tau)) + fixedCost;
% Hot start-up cost - size 10 1 
hotStartUpCost = hotAndFuelCost .* minDownTime + fixedCost;
% Total start-up cost
totalStartUpCost = coolStartUpCost + hotStartUpCost;

% Demands
Hor = ones(HorizonLen,1); 
Hor = [Hor;Hor];
Dem = Loads*Hor';


% Definition of variables 
% x -- [g;z;y] -- g -- [g1,.,gI] -- gi -- [gi1,.,giT] -- same for z and y


% x -- [g_t,g_h;z_t;y_t] -- g_ti -- [g_ti1,.,g_tiT] -- same for z_t and y_t

% Problem Formulation
% Initial numbers and dimension


I = NumThermal + NumHydro; 
T = HorizonLen;                            % Number of generators and time steps
d = sum(Dem)';                             % Demand over T periods
Q = [maxPower; 0.3*volumeToPower.*maxFlood];
q = [minPower; 0.3*volumeToPower.*minFlood];         % Maximum and Minimum Generation capacities
n = 3*I*T;                                 % total number of decision variables in the optimization problem
mbin = 2*I*T;                              % total number of binary variables in the problem
L = [minUpTime; inf(NumHydro,1)];          % for hydro generators no up and down time constraits
l = [minDownTime; zeros(NumHydro,1)];


% Define scaling factor -- Preconditioner 
Fscal = mean(q + 0.5*(Q-q));

% Definition of variables 
% x -- [g;z;y] -- g -- [g1,.,gI] -- gi -- [gi1,.,giT] -- same for z and y


% x -- [g_t,g_h;z_t;z_h;y_t;y_h] -- g_t -- [g_t1,.,g_tI_t] -- g_ti -- [g_ti1,.,g_tiT] -- same for z_t and y_t

%assuming renewable energy has no production cost
quadraticCoefficients = [quadraticCoefficients; zeros(NumHydro,1)]; 
linearCoefficients = [linearCoefficients; zeros(NumHydro,1)];
constantCoefficients = [constantCoefficients; zeros(NumHydro,1)];
totalStartUpCost = [totalStartUpCost; zeros(NumHydro,1)];

% Define the quadratic cost function - dimensions 3I*T x 1
quad = [reshape(repmat(quadraticCoefficients(1:I),1,T)',I*T,1); zeros(2*I*T,1)]; 

% Define the linear cost function - dimensions I*T x 1
lin =  [reshape(repmat(linearCoefficients(1:I),1,T)',I*T,1); zeros(I*T,1); ...
        reshape(repmat(totalStartUpCost(1:I),1,T)',I*T,1)]; 
% Define the constant cost function - dimensions I*T x 1
const = [reshape(repmat(constantCoefficients(1:I),1,T)',I*T,1); zeros(2*I*T,1)];  


% Definition of inequalities in the form Ax <= b
    
% Demand constraints -- Expression (8)
A1  =  [-repmat(eye(T),1,I), zeros(T,2*I*T)];
% A1  =  [-repmat(eye(T),1,NumThermal), zeros(T,T*NumHydro), zeros(T,2*I*T)]; % dimensions T x (3*I*T) 

dumm1 = []; dumm2 = [];
   for i = 1:I % i = 1:NumThermal
      dumm1 = blkdiag(dumm1,((Q(i,1))/Fscal)*eye(T));
      dumm2 = blkdiag(dumm2,((q(i,1))/Fscal)*eye(T));
   end

A2  = [eye(I * T), -dumm1, zeros(I*T)]; % Capacity upper bnd cons - dimensions I*T x (3*I*T)

A3  = [-eye(I * T), dumm2, zeros(I*T)]; % Capacity lower bnd cons - dimensions I*T x (3*I*T)

clear dumm1 dumm2;

dumm3 =  [-eye(T-1), zeros(T-1,1)] + [zeros(T-1,1), eye(T-1)];
dumm4 =  [zeros(T-1,1), eye(T-1)];
dumm5 =  [];
dumm6 =  [];
for i = 1:I % i = 1:NumThermal
    dumm5 = blkdiag(dumm5,dumm3);
    dumm6 = blkdiag(dumm6,dumm4);
end


% Expression (5) -- on/off constraints
% Startup and Running vars related by A4

A4 = [zeros(I*(T-1),I*T), dumm5, -dumm6];

In = []; In2 = [];
for i = 1:I
    dumm8 = []; dumm10 = [];
    for t = 2:T   
        % Indices -- first set
        numind = min(L(i),T-t+1); ind = t:t+numind-1;
        dumm7 = zeros(min(L(i),T-t+1),T);
        dumm7(1:numind,ind) = -eye(numind);
        dumm7(:,t-1) = -1; dumm7(:,t) = 1; dumm7(1,t) = 0;
        dumm8 = [dumm8;dumm7];
        % Indices -- second set
        numind2 = min(l(i),T-t+1); ind2 = t:t+numind2-1;
        dumm9 = zeros(min(l(i),T-t+1),T);
        dumm9(1:numind,ind) = -eye(numind);
        dumm9(:,t-1) = -1; dumm9(:,t) = 1; dumm9(1,t) = 0;
        dumm10 = [dumm10;dumm9];
    end
    In  = blkdiag(In,dumm8);
    In2 = blkdiag(In2,dumm10);
end


% Expression (6) refers to up and down-time constraints 
In  = [zeros(size(In,1),I*T), In, zeros(size(In,1),I*T)];  % Up time cons 
In2 = [zeros(size(In2,1),I*T), In2, zeros(size(In2,1),I*T)]; % Down time cons 


% Remove 0 row entries from In - ideally not required
dropsin  = find(sum(In,2)==0);    In(dropsin,:) = [];
dropsin2 = find(sum(In2,2)==0);   In2(dropsin2,:) = [];


% In matrix form
A   = [A1;A2;A3;A4;In;-In2];
b   = [-((d(1:T,1))/Fscal);zeros(2*I*T,1); zeros(size(A4,1),1); ...
%    zeros(size(In,1),1)];
    zeros(size(In,1),1); ones(size(In2,1),1)];  

m = size(A,1);
options = optimset; options.Display = 'on'; options.MaxIter = 1e4;
warning off;
    
% Rescale both QQ and lin
% mbin = ceil(n/10); Fscal = 1;
nx = n - mbin;
QQ = diag([(Fscal^2)*quad(1:nx,1);quad(nx+1:end)]); %+1e-3*eye(n);
%randn('state',lpo)


%QQ = randn(n); QQ =1e4*(QQ*QQ');
lin = [Fscal*lin(1:nx,1);lin(nx+1:end)];

% sulfur emission coefficients 
%alpha_s = [198.33; 195.34; 155.15; 152.26; 152.26; 101.43; 111.87; 126.62; ...
%           134.15; 142.26; 198.33; 195.34; 155.15; 152.26; 152.26; 101.43; 111.87;...
%           126.62; 134.15; 142.26];
beta_s = [2.06; 2.09; 2.14; 2.25; 2.11; 3.45; 2.62; 5.18; 5.38; 5.40; ... 
          2.06; 2.09; 2.14; 2.25; 2.11; 3.45; 2.62; 5.18; 5.38; 5.40];
gamma_s = [0.00019; 0.00018; 0.00220; 0.00220; 0.00210; 0.00250; ...
           0.00220; 0.00420; 0.00540; 0.00550; 0.00019; 0.00018; ...
           0.00220; 0.00220; 0.00210; 0.00250; 0.00220; 0.00420; 0.00540; 0.00550];

%CO2 emission coefficients 
%alpha_c = [130.00; 132.00; 137.70; 130.00; 125.00; 110.00; 135.00; 157.00; ...
%           160.00; 137.70; 130.00; 132.00; 137.70; 130.00; 125.00; 110.00; ... 
%           135.00; 157.00; 160.00; 137.70];
beta_c = [-2.86; -2.72; -2.94; -2.35; -2.36; -2.28; -2.36; -1.29; -1.14; -2.14; ...
          -2.86; -2.72; -2.94; -2.35; -2.36; -2.28; -2.36; -1.29; -1.14; -2.14];
gamma_c = [0.022; 0.020; 0.044; 0.058; 0.065; 0.080; 0.075; 0.082; 0.090; 0.084; ...
           0.022; 0.020; 0.044; 0.058; 0.065; 0.080; 0.075; 0.082; 0.090; 0.084];

% replicate each element of coeffs T times to expand for each timestep
%alpha_s = repelem(alpha_s, T);
beta_s = repelem(beta_s, T);
gamma_s = repelem(gamma_s, T);
%alpha_c = repelem(alpha_c, T);
beta_c = repelem(beta_c, T);
gamma_c = repelem(gamma_c, T);

% Combine coefficients for third quadratic objective 
% and append zeros to match the size n
%alpha = [alpha_s + alpha_c; zeros(T * (3 * NumHydro + 2 * NumThermal), 1)];
beta = [beta_s + beta_c; zeros(T * (3 * NumHydro + 2 * NumThermal), 1)];
gamma = [gamma_s + gamma_c; zeros(T * (3 * NumHydro + 2 * NumThermal), 1)];
lin2 = beta;

% Set Gurobi parameters 
params = struct();
params.OutputFlag = 1;  % Display output (controls the amount of output displayed during the optimization process)
params.DisplayInterval = 1; % controls how often intermediate output is displayed

% lover and upper bounds for f1 
l1 = 3009350.196; 
u1 = 3233076.8277;% 3305171.5106; % * 3233076.8277; %3306171.5106;

% Define epsilon values
epsilons_f1 = 0:0.1:1;

% Initialize solutions
solutions_f2 = cell(length(epsilons_f1), 1);

clear model_f2;

    % Define the model for f2
model_f2.modelsense = 'min';  
model_f2.A = sparse(A);  % Convert A matrix to sparse format
model_f2.rhs = b;  % Right-hand side of the constraints
model_f2.vtype = [repmat('C', nx, 1); repmat('B', mbin, 1)];  % nx continuous, mbin binary variables
model_f2.sense = '<';

Q2 = sparse(double(diag(gamma)));
model_f2.multiobj(1).objn = lin2;  % Linear coefficients of the objective function
model_f2.multiobj(1).Q = Q2; % Quadratic coefficients of the objective function
model_f2.multiobj(1).weight = 1.0;
gurobi_t = 0;
Q1 = sparse(double(QQ));


% Iterate over epsilon values for f1
for i = 1:length(epsilons_f1)
    epsilon = epsilons_f1(i);
    
    clear model_f2.quadcon(1);
    % Add epsilon constraint for f2
    model_f2.quadcon(1).Qc = Q1;
    model_f2.quadcon(1).q = lin;
    model_f2.quadcon(1).rhs = l1 + epsilon*(u1-l1);
    model_f2.quadcon(1).sense = '<';
    model_f2.quadcon(1).name = ['epsilon_constraint_f2_', num2str(i)];
    
    % Solve the model
    st = cputime;
    solutions_f2{i} = gurobi(model_f2, params);
    et = cputime-st;
    gurobi_t = et+gurobi_t;
end

total_t = cputime-ts;

% Display solutions for f2
disp('Solutions for f2:');
for i = 1:length(epsilons_f1)
    epsilon = epsilons_f1(i);
    solution = solutions_f2{i};
    
    disp(['Epsilon = ', num2str(epsilon)]);
    if strcmp(solution.status, 'OPTIMAL')
        disp(['  Status: ', solution.status]);
        disp(['  Objective value f2: ', num2str(solution.objval)]);
        bound = l1 + epsilon*(u1-l1);
        disp(['  upper bound for f1: ', num2str(bound)]);
    elseif strcmp(solution.status, 'INFEASIBLE')
        disp('  Status: Infeasible');
    else
        disp(['  Status: ', solution.status]);
        disp('  No solution found.');
    end
    disp(' ');
end

fprintf('Total runtime: %.2f seconds\n', total_t);
fprintf('Gurobi runtime: %.2f seconds\n', gurobi_t);
