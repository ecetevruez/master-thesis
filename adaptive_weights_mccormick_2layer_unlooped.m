ts = cputime;
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

% Demands (=Loads)
Hor = ones(HorizonLen,1); 
Hor = [Hor;Hor];
Dem = Loads*Hor';


% Definition of variables 
% x -- [g;z;y] -- g -- [g1,.,gI] -- gi -- [gi1,.,giT] -- same for z and y


% x -- [g_t,g_h;z_t;y_t] -- g_ti -- [g_ti1,.,g_tiT] -- same for z_t and y_t

% Problem Formulation
% Initial numbers and dimension

I = NumThermal + NumHydro; 
T = HorizonLen;                                      % Number of generators and time steps
d = sum(Dem)';                                       % Demand over T periods
Q = [maxPower; 0.3*volumeToPower.*maxFlood];
q = [minPower; 0.3*volumeToPower.*minFlood];         % Maximum and Minimum Generation capacities
n = 3*I*T;                                           % total number of decision variables in the optimization problem
mbin = 2*I*T;                                        % total number of binary variables in the problem
L = [minUpTime; zeros(NumHydro,1)];                  % for hydro generators no up and down time constraits
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
% does not play a role, just a constant to the objective
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

% Generation Capacity constraints -- Expression (7) from the paper

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
    for t = 2:T    % for t = 2:NumThermal
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
% A   = [A1;A2;A3;A4;In];
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

beta_s = [2.06; 2.09; 2.14; 2.25; 2.11; 3.45; 2.62; 5.18; 5.38; 5.40; ... 
          2.06; 2.09; 2.14; 2.25; 2.11; 3.45; 2.62; 5.18; 5.38; 5.40];
gamma_s = [0.00019; 0.00018; 0.00220; 0.00220; 0.00210; 0.00250; ...
           0.00220; 0.00420; 0.00540; 0.00550; 0.00019; 0.00018; ...
           0.00220; 0.00220; 0.00210; 0.00250; 0.00220; 0.00420; 0.00540; 0.00550];


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

%alpha = [alpha_s + alpha_c; zeros(T * (3 * NumHydro + 2 * NumThermal), 1)];
beta = [beta_s + beta_c; zeros(T * (3 * NumHydro + 2 * NumThermal), 1)];
gamma = [gamma_s + gamma_c; zeros(T * (3 * NumHydro + 2 * NumThermal), 1)];
lin2 = beta;

clear model;
%Construct Gurobi model
model.modelsense = 'min';  
model.A = sparse(A);  % Convert A matrix to sparse format
model.rhs = b;  % Right-hand side of the constraints
model.vtype = [repmat('C', nx, 1); repmat('B', mbin, 1)];  % nx continuous, mbin binary variables
model.sense = '<';

% First objective (cost) 
Q1 = sparse(double(QQ));
model.multiobj(1).objn = lin;  % Linear coefficients of the objective function
model.multiobj(1).Q = Q1; % Quadratic coefficients of the objective function
model.multiobj(1).weight = 1.0;

% Second objective (emission) 
Q2 = sparse(double(diag(gamma)));
model.multiobj(2).objn = lin2;
model.multiobj(2).Q = Q2;
model.multiobj(2).weight = 1.0;

% Save model 
% gurobi_write(model, 'your_model.lp');

% Set Gurobi parameters 
params = struct();
params.OutputFlag = 1;  % Display output (controls the amount of output displayed during the optimization process)
params.DisplayInterval = 1; % controls how often intermediate output is displayed

% regularize Q1 and Q2
average_value = sum(Q1(:)) / numel(Q1);
n1 = size(Q1, 1); 
epsilon1 = average_value * 0.01;
Q1_reg = Q1 + (epsilon1 * eye(n1)); % Regularize Q1 by adding epsilon times the identity matrix

average_value2 = sum(Q2(:)) / numel(Q2);
n2 = size(Q2, 1); 
epsilon2 = average_value2 * 0.01;
Q2_reg = Q2 + (epsilon2 * eye(n2)); % Regularize Q2 by adding epsilon times the identity matrix

L1 = chol(Q1_reg, 'lower');
L2 = chol(Q2_reg, 'lower');

% lower and upper bounds for variables
x_U = [repelem(maxPower, T); repelem(0.3 * volumeToPower .* maxFlood, T); ones(2*I*T, 1)];
x_L = [zeros(3*I*T, 1)];

y1_U = L1'*x_U;
y1_L = L1'*x_L;

y2_U = L2'*x_U;
y2_L = L2'*x_L;

M = 1e9;

y1_a_L = y1_L;
y1_a_U = (y1_U+y1_L)/2;
y1_b_L = y1_a_U;
y1_b_U = y1_U;

y2_a_L = y2_L;
y2_a_U = (y2_U+y2_L)/2;
y2_b_L = y2_a_U;
y2_b_U = y2_U;

gurobi_t = 0;

% Step 2
% Perform multiobjective optimization using
% the usual weighted-sum approach with a small number of divisions
ninitial = 10;

% Initialize results array
results = struct('lambda', zeros(ninitial, 1), 'obj_values', zeros(ninitial, 1), 'x_values', cell(ninitial, 1));

for i = 1:ninitial
    % Define lambda
    lambda = (i - 1) / (ninitial - 1);

    % Set weights in the Gurobi model
    model.multiobj(1).weight = lambda;
    model.multiobj(2).weight = 1 - lambda;

    % Optimize
    st = cputime;    
    result = gurobi(model, params);
    et = cputime-st;
    gurobi_t = et+gurobi_t;

    % Store results
    results(i).lambda = lambda;
    results(i).obj_values = result.objval(:)';
    results(i).x_values = result.x;
end

% Display the updated results
disp('Results after the initial 10:');
disp('Lambda | Objective Values');
disp('-------|-----------------');
for i = 1:numel(results)
    disp([num2str(results(i).lambda) '    | ' num2str(results(i).obj_values)]);
end

% Define a function to compute Euclidean distance between two solutions
euclidean_distance = @(x1, x2) norm(x1 - x2);
tic;

countit = 0;  

    %Step 3
    % Calculate the length of each segment before removing overlapping
    % solutions
segment_lengths = zeros(numel(results) - 1, 1);
for i = 1:numel(results) - 1
    segment_lengths(i) = euclidean_distance(results(i).obj_values, results(i + 1).obj_values);
end

    % Compute the lengths of the segments between all 
    % the neighboring solutions. Delete nearly overlapping solutions.

    % Prescribed distance epsilon calculated based on the average objective values of the results
average_obj_values = mean([results.obj_values]); 

epsilon = 1e-3 * mean(average_obj_values);

    % Compute distances between neighboring solutions and remove nearly overlapping solutions
distances = zeros(ninitial - 1, 1);
indices_to_delete = [];
for i = 1:ninitial - 1
    distances(i) = euclidean_distance(results(i).obj_values, results(i + 1).obj_values);
    if distances(i) < epsilon
        indices_to_delete = [indices_to_delete, i + 1]; 
    end
end

    % Delete nearly overlapping solutions
results(indices_to_delete) = [];

    % Display the updated results
   % Check if there is more than one result before displaying
%if numel(results) > 1
    %disp('Results after removing nearly overlapping solutions:');
    %disp('Lambda | Objective Values');
    %disp('-------|-----------------');
    %for i = 1:numel(results)
    %    disp([num2str(results(i).lambda) '    | ' num2str(results(i).obj_values)]);
    %end

%end
    % Step 5
if numel(results) == 1
    total_t = cputime-ts;
    disp('Adaptive weights method completed0:');
    disp(['Total iterations: ' num2str(countit)]);
    disp('Objective Values:');
    fprintf('%14.6f   %14.6f\n', results.obj_values);
    fprintf('Total runtime: %.2f seconds\n', total_t);
    fprintf('Gurobi runtime: %.2f seconds\n', gurobi_t);  
    return;
end

    % Step 4
    % Determine the number of further refinements
    % in each of the regions

    % Calculate the average length of all segments
average_length = mean(segment_lengths);

    % Set the constant C 
C = 1.1;

    % Determine the number of further refinements for each segment
num_refinements = round(C * segment_lengths / average_length);

    % Display the number of further refinements for each segment
    %disp('Number of Further Refinements:');
    %disp('Segment | Num Refinements');
    %disp('--------|-----------------');
    %for i = 1:numel(num_refinements)
    %    disp([num2str(i) '       | ' num2str(num_refinements(i))]);
    %end

    % Determine the offset distances and compute final Pareto front
nsegments = numel(results) - 1;
offset_distances = zeros(nsegments, 2);

    % Step 6
    % scaling factor for delta_J
scaling_factor = 0.78; % for scaling_factor = 0.5 Total iterations: 9

for i = 1:nsegments
        % Calculate angle θ
    P1 = results(i).obj_values;
    P2 = results(i + 1).obj_values;
    theta = atan(-(P1(2) - P2(2)) / (P1(1) - P2(1)));

        % Determine offset distances with scaling factor
    delta_J = scaling_factor * max(distances);
    delta_1 = delta_J * cos(theta);
    delta_2 = delta_J * sin(theta);

        % Store offset distances
    offset_distances(i, :) = [delta_1, delta_2];
end
    
%disp('Offset Distances:');
%disp('Segment | Delta_J1 | Delta_J2');
%disp('--------|----------|----------');
%for i = 1:nsegments
%    disp([num2str(i) '       | ' num2str(offset_distances(i, 1)) '        | ' num2str(offset_distances(i, 2))]);
%end

    % Step 7: Impose additional inequality constraints and
    % conduct sub-optimization with the weighted-sum method
    % in each of the feasible regions
    
    % submodel with linearized additional constraints
%A_ext = [A, zeros(size(A, 1),6*n); ...                                                     % original constraints
%         lin', zeros(1, 2*n), 0.5*ones(1, n), zeros(1, n), zeros(1,2*n); ...                                      % J1 <= P1 - delta1
%         lin2', zeros(1, 2*n), zeros(1, n), 0.5*ones(1, n), zeros(1,2*n); ...                                     % J2 <= P2 - delta2
%         zeros(n, n), 2 * diag(y1_a_L), zeros(n, n), -eye(n), zeros(n, n), M*eye(n), zeros(n,n); ...         % underestimator 1 obj 1 
%         zeros(n, n), 2 * diag(y1_a_U), zeros(n, n), -eye(n), zeros(n, n), M*eye(n), zeros(n,n); ...         % underestimator 2 obj 1
%         zeros(n, n), diag(-(y1_a_U+y1_a_L)), zeros(n, n), eye(n), zeros(n, n), M*eye(n), zeros(n,n); ...    % overestimator  1 obj 1 
%         zeros(n, n), zeros(n,n), 2 * diag(y2_a_L), zeros(n, n), -eye(n), M*eye(n), zeros(n,n); ...          % underestimator 1 obj 2
%         zeros(n, n), zeros(n,n), 2 * diag(y2_a_U), zeros(n, n), -eye(n), M*eye(n), zeros(n,n); ...          % underestimator 2 obj 2
%         zeros(n, n), zeros(n,n), diag(-(y2_a_U+y2_a_L)), zeros(n, n), eye(n), M*eye(n), zeros(n,n); ...     % overestimator  1 obj 2 
%         zeros(n, n), 2 * diag(y1_b_L), zeros(n, n), -eye(n), zeros(n, n), zeros(n,n), M*eye(n); ...         % underestimator 1 obj 1 
%         zeros(n, n), 2 * diag(y1_b_U), zeros(n, n), -eye(n), zeros(n, n), zeros(n,n), M*eye(n); ...         % underestimator 2 obj 1
%         zeros(n, n), diag(-(y1_b_U+y1_b_L)), zeros(n, n), eye(n), zeros(n, n), zeros(n,n), M*eye(n); ...    % overestimator  1 obj 1 
%         zeros(n, n), zeros(n,n), 2 * diag(y2_b_L), zeros(n, n), -eye(n), zeros(n,n), M*eye(n); ...          % underestimator 1 obj 2
%         zeros(n, n), zeros(n,n), 2 * diag(y2_b_U), zeros(n, n), -eye(n), zeros(n,n), M*eye(n); ...          % underestimator 2 obj 2
%         zeros(n, n), zeros(n,n), diag(-(y2_b_U+y2_b_L)), zeros(n, n), eye(n), zeros(n,n), M*eye(n); ...     % overestimator  1 obj 2 
%         L1', -eye(n), zeros(n,3*n), zeros(n,2*n); ...                                                            % L1'*x - y1 = 0
%         L2', zeros(n,n), -eye(n), zeros(n,2*n), zeros(n,2*n); ...                                                % L2'*x - y2 = 0
%         zeros(n,5*n), eye(n), eye(n) ...                                                                                 % q_a + q_b = 1                         
%         ];

% Initialize the size of the matrix
numRows = size(A, 1) + 2 + 15 * n;
numCols = size(A, 2) + 6 * n;

% Initialize a sparse matrix
A_ext = sparse(numRows, numCols);

% Fill the original constraints
A_ext(1:size(A, 1), 1:size(A, 2)) = A;

% J1 <= P1 - delta1
A_ext(size(A, 1) + 1, :) = [lin', zeros(1, 2 * n), ones(1, n), zeros(1, n), zeros(1, 2 * n)];

% J2 <= P2 - delta2
A_ext(size(A, 1) + 2, :) = [lin2', zeros(1, 2 * n), zeros(1, n), ones(1, n), zeros(1, 2 * n)];

% Underestimator 1 obj 1
startRow = size(A, 1) + 3;
A_ext(startRow:startRow + n - 1, :) = [zeros(n, n), 2 * diag(y1_a_L), zeros(n, n), -eye(n), zeros(n, n), M * eye(n), zeros(n, n)];

% Underestimator 2 obj 1
startRow = startRow + n;
A_ext(startRow:startRow + n - 1, :) = [zeros(n, n), 2 * diag(y1_a_U), zeros(n, n), -eye(n), zeros(n, n), M * eye(n), zeros(n, n)];

% Overestimator 1 obj 1
startRow = startRow + n;
A_ext(startRow:startRow + n - 1, :) = [zeros(n, n), diag(-(y1_a_U + y1_a_L)), zeros(n, n), eye(n), zeros(n, n), M * eye(n), zeros(n, n)];

% Underestimator 1 obj 2
startRow = startRow + n;
A_ext(startRow:startRow + n - 1, :) = [zeros(n, n), zeros(n, n), 2 * diag(y2_a_L), zeros(n, n), -eye(n), M * eye(n), zeros(n, n)];

% Underestimator 2 obj 2
startRow = startRow + n;
A_ext(startRow:startRow + n - 1, :) = [zeros(n, n), zeros(n, n), 2 * diag(y2_a_U), zeros(n, n), -eye(n), M * eye(n), zeros(n, n)];

% Overestimator 1 obj 2
startRow = startRow + n;
A_ext(startRow:startRow + n - 1, :) = [zeros(n, n), zeros(n, n), diag(-(y2_a_U + y2_a_L)), zeros(n, n), eye(n), M * eye(n), zeros(n, n)];

% Underestimator 1 obj 1
startRow = startRow + n;
A_ext(startRow:startRow + n - 1, :) = [zeros(n, n), 2 * diag(y1_b_L), zeros(n, n), -eye(n), zeros(n, n), zeros(n, n), M * eye(n)];

% Underestimator 2 obj 1
startRow = startRow + n;
A_ext(startRow:startRow + n - 1, :) = [zeros(n, n), 2 * diag(y1_b_U), zeros(n, n), -eye(n), zeros(n, n), zeros(n, n), M * eye(n)];

% Overestimator 1 obj 1
startRow = startRow + n;
A_ext(startRow:startRow + n - 1, :) = [zeros(n, n), diag(-(y1_b_U + y1_b_L)), zeros(n, n), eye(n), zeros(n, n), zeros(n, n), M * eye(n)];

% Underestimator 1 obj 2
startRow = startRow + n;
A_ext(startRow:startRow + n - 1, :) = [zeros(n, n), zeros(n, n), 2 * diag(y2_b_L), zeros(n, n), -eye(n), zeros(n, n), M * eye(n)];

% Underestimator 2 obj 2
startRow = startRow + n;
A_ext(startRow:startRow + n - 1, :) = [zeros(n, n), zeros(n, n), 2 * diag(y2_b_U), zeros(n, n), -eye(n), zeros(n, n), M * eye(n)];

% Overestimator 1 obj 2
startRow = startRow + n;
A_ext(startRow:startRow + n - 1, :) = [zeros(n, n), zeros(n, n), diag(-(y2_b_U + y2_b_L)), zeros(n, n), eye(n), zeros(n, n), M * eye(n)];

% L1'*x - y1 = 0
startRow = startRow + n;
A_ext(startRow:startRow + n - 1, :) = [L1', -eye(n), zeros(n, 3 * n), zeros(n, 2 * n)];

% L2'*x - y2 = 0
startRow = startRow + n;
A_ext(startRow:startRow + n - 1, :) = [L2', zeros(n, n), -eye(n), zeros(n, 2 * n), zeros(n, 2 * n)];

% q_a + q_b = 1
startRow = startRow + n;
A_ext(startRow:startRow + n - 1, :) = [zeros(n, 5 * n), eye(n), eye(n)];


b_ext = [b; ...
         P1(1) - delta_1; ...
         P2(2) - delta_2; ...
         (y1_a_L .* y1_a_L) + M * ones(n,1); ...   
         y1_a_U .* y1_a_U + M * ones(n,1); ...   
         -(y1_a_U .* y1_a_L) + M * ones(n,1); ... 
         y2_a_L .* y2_a_L + M * ones(n,1); ...   
         y2_a_U .* y2_a_U + M * ones(n,1); ...   
         -(y2_a_U .* y2_a_L)+ M * ones(n,1); ...  
         (y1_b_L .* y1_b_L) + M * ones(n,1); ...   
         y1_b_U .* y1_b_U + M * ones(n,1); ...   
         -(y1_b_U .* y1_b_L) + M * ones(n,1); ... 
         y2_b_L .* y2_b_L + M * ones(n,1); ...   
         y2_b_U .* y2_b_U + M * ones(n,1); ...   
         -(y2_b_U .* y2_b_L)+ M * ones(n,1); ...  
         zeros(2*n,1); ...   
         ones(n,1)
         ];

    % Initialize all constraints as '<'
    % Set the sense for the last 2*n constraints as '=' 
sense = repmat('<', size(A_ext, 1), 1);  
sense(end-2*n+1:end) = '=';

clear submodel;

    %Construct Gurobi model
submodel.modelsense = 'min';  
submodel.A = sparse(A_ext);  % Convert A matrix to sparse format
submodel.rhs = b_ext;        % Right-hand side of the constraints
submodel.vtype = [repmat('C', nx, 1); repmat('B', mbin, 1); repmat('C', 4*n, 1); repmat('B', 2*n, 1)];  % nx continuous, mbin binary variables
submodel.sense = sense;
submodel.lb = [x_L; -inf*ones(6*n,1)]; 
submodel.ub = [x_U; inf*ones(6*n,1)];
Q1_ext = zeros(7*n, 7*n);  % Create an extended matrix filled with zeros
Q1_ext(1:n, 1:n) = Q1; 
Q2_ext = zeros(7*n, 7*n);  % Create an extended matrix filled with zeros
Q2_ext(1:n, 1:n) = Q2; 

submodel.multiobj(1).objn = [lin; zeros(6*n,1)];  
submodel.multiobj(1).Q = Q1_ext; 

submodel.multiobj(2).objn = [lin2; zeros(6*n,1)];
submodel.multiobj(2).Q = Q2_ext;
  
    % number refinements for segments that require further refinements
k = num_refinements((num_refinements ~= 0));
    
    % Break out of the loop if there are no segments requiring refinement
%if isempty(k)
%    total_t = cputime-ts;
%    disp('Adaptive weights method completed (k):');
%    disp(['Total iterations: ' num2str(iteration)]);
%    disp('Objective Values:');
%    fprintf('%14.6f   %14.6f\n', results.obj_values);

% Display elapsed time
%    fprintf('Total runtime: %.2f seconds\n', total_t);
%    fprintf('Gurobi runtime: %.2f seconds\n', gurobi_t);    
%    return;
%end

for i = 1:k
        % Define lambda
    lambda = (i - 1) / (k - 1);

        % Set weights in the Gurobi model
    submodel.multiobj(1).weight = lambda;
    submodel.multiobj(2).weight = 1 - lambda;

        % Optimize
    st = cputime;    
    result = gurobi(submodel, params);
    et = cputime-st;
    gurobi_t = et+gurobi_t;

        % Store results
    results(i).lambda = lambda;
    results(i).obj_values = result.objval(:)';
    results(i).x_values = result.x;
end
countit = countit + 1; % count iteration 


% Display the updated results
disp('Results after 1');
disp('Lambda | Objective Values');
disp('-------|-----------------');
for i = 1:numel(results)
    disp([num2str(results(i).lambda) '    | ' num2str(results(i).obj_values)]);
end

%save("adaptive_weights_mccormic.mat")
segment_lengths = zeros(numel(results) - 1, 1);
for i = 1:numel(results) - 1
    segment_lengths(i) = euclidean_distance(results(i).obj_values, results(i + 1).obj_values);
end

    % Compute the lengths of the segments between all 
    % the neighboring solutions. Delete nearly overlapping solutions.

    % Prescribed distance epsilon calculated based on the average objective values of the results
average_obj_values = mean([results.obj_values]); 

epsilon = 1e-3 * mean(average_obj_values);

    % Compute distances between neighboring solutions and remove nearly overlapping solutions
distances = zeros(ninitial - 1, 1);
indices_to_delete = [];
for i = 1:ninitial - 1
    distances(i) = euclidean_distance(results(i).obj_values, results(i + 1).obj_values);
    if distances(i) < epsilon
        indices_to_delete = [indices_to_delete, i + 1]; 
    end
end

    % Delete nearly overlapping solutions
results(indices_to_delete) = [];

    % Display the updated results
   % Check if there is more than one result before displaying
%if numel(results) > 1
    %disp('Results after removing nearly overlapping solutions:');
    %disp('Lambda | Objective Values');
    %disp('-------|-----------------');
    %for i = 1:numel(results)
    %    disp([num2str(results(i).lambda) '    | ' num2str(results(i).obj_values)]);
    %end

%end
    % Step 5
if numel(results) == 1
    total_t = cputime-ts;
    disp('Adaptive weights method completed1:');
    disp(['Total iterations: ' num2str(countit)]);
    disp('Objective Values:');
    fprintf('%14.6f   %14.6f\n', results.obj_values);
    fprintf('Total runtime: %.2f seconds\n', total_t);
    fprintf('Gurobi runtime: %.2f seconds\n', gurobi_t);   
    return
end


    % Calculate the average length of all segments
average_length = mean(segment_lengths);

    % Set the constant C 
C = 1.1;

    % Determine the number of further refinements for each segment
num_refinements = round(C * segment_lengths / average_length);

    % Display the number of further refinements for each segment
    %disp('Number of Further Refinements:');
    %disp('Segment | Num Refinements');
    %disp('--------|-----------------');
    %for i = 1:numel(num_refinements)
    %    disp([num2str(i) '       | ' num2str(num_refinements(i))]);
    %end

    % Determine the offset distances and compute final Pareto front
nsegments = numel(results) - 1;
offset_distances = zeros(nsegments, 2);

    % Step 6
    % scaling factor for delta_J
scaling_factor = 0.48; % for scaling_factor = 0.5 Total iterations: 9

for i = 1:nsegments
        % Calculate angle θ
    P1 = results(i).obj_values;
    P2 = results(i + 1).obj_values;
    theta = atan(-(P1(2) - P2(2)) / (P1(1) - P2(1)));

        % Determine offset distances with scaling factor
    delta_J = scaling_factor * max(distances);
    delta_1 = delta_J * cos(theta);
    delta_2 = delta_J * sin(theta);

        % Store offset distances
    offset_distances(i, :) = [delta_1, delta_2];
end
    
%disp('Offset Distances:');
%disp('Segment | Delta_J1 | Delta_J2');
%disp('--------|----------|----------');
%for i = 1:nsegments
%    disp([num2str(i) '       | ' num2str(offset_distances(i, 1)) '        | ' num2str(offset_distances(i, 2))]);
%end

    % Step 7: Impose additional inequality constraints and
    % conduct sub-optimization with the weighted-sum method
    % in each of the feasible regions
    
    % submodel with linearized additional constraints


b_ext = [b; ...
         P1(1) - delta_1; ...
         P2(2) - delta_2; ...
         (y1_a_L .* y1_a_L) + M * ones(n,1); ...   
         y1_a_U .* y1_a_U + M * ones(n,1); ...   
         -(y1_a_U .* y1_a_L) + M * ones(n,1); ... 
         y2_a_L .* y2_a_L + M * ones(n,1); ...   
         y2_a_U .* y2_a_U + M * ones(n,1); ...   
         -(y2_a_U .* y2_a_L)+ M * ones(n,1); ...  
         (y1_b_L .* y1_b_L) + M * ones(n,1); ...   
         y1_b_U .* y1_b_U + M * ones(n,1); ...   
         -(y1_b_U .* y1_b_L) + M * ones(n,1); ... 
         y2_b_L .* y2_b_L + M * ones(n,1); ...   
         y2_b_U .* y2_b_U + M * ones(n,1); ...   
         -(y2_b_U .* y2_b_L)+ M * ones(n,1); ...  
         zeros(2*n,1); ...   
         ones(n,1)
         ];

    % Initialize all constraints as '<'
    % Set the sense for the last 2*n constraints as '=' 
sense = repmat('<', size(A_ext, 1), 1);  
sense(end-2*n+1:end) = '=';

clear submodel;

    %Construct Gurobi model
submodel.modelsense = 'min';  
submodel.A = sparse(A_ext);  % Convert A matrix to sparse format
clear A_ext;
submodel.rhs = b_ext;        % Right-hand side of the constraints
submodel.vtype = [repmat('C', nx, 1); repmat('B', mbin, 1); repmat('C', 4*n, 1); repmat('B', 2*n, 1)];  % nx continuous, mbin binary variables
submodel.sense = sense;
submodel.lb = [x_L; -inf*ones(6*n,1)]; 
submodel.ub = [x_U; inf*ones(6*n,1)];

%Q1_ext = zeros(5*n, 5*n);  % Create an extended matrix filled with zeros
%Q1_ext(1:n, 1:n) = Q1; 
%Q2_ext = zeros(5*n, 5*n);  % Create an extended matrix filled with zeros
%Q2_ext(1:n, 1:n) = Q2; 

submodel.multiobj(1).objn = [lin; zeros(6*n,1)];  
submodel.multiobj(1).Q = Q1_ext; 

submodel.multiobj(2).objn = [lin2; zeros(6*n,1)];
submodel.multiobj(2).Q = Q2_ext;
  
    % number refinements for segments that require further refinements
k = num_refinements((num_refinements ~= 0));
    
    % Break out of the loop if there are no segments requiring refinement
%if isempty(k)
%    total_t = cputime-ts;
%    disp('Adaptive weights method completed (k):');
%    disp(['Total iterations: ' num2str(iteration)]);
%    disp('Objective Values:');
%    fprintf('%14.6f   %14.6f\n', results.obj_values);
% Display elapsed time
%    fprintf('Total runtime: %.2f seconds\n', total_t);
%    fprintf('Gurobi runtime: %.2f seconds\n', gurobi_t);    
%    return;
%end

for i = 1:k
        % Define lambda
    lambda = (i - 1) / (k - 1);

        % Set weights in the Gurobi model
    submodel.multiobj(1).weight = lambda;
    submodel.multiobj(2).weight = 1 - lambda;

        % Optimize
    st = cputime;    
    result = gurobi(submodel, params);
    et = cputime-st;
    gurobi_t = et+gurobi_t;

        % Store results
    results(i).lambda = lambda;
    results(i).obj_values = result.objval(:)';
    results(i).x_values = result.x;
end
countit = countit + 1; % count iteration 


% Display the updated results
%disp('Results after 2:');
%disp('Lambda | Objective Values');
%disp('-------|-----------------');
%for i = 1:numel(results)
%    disp([num2str(results(i).lambda) '    | ' num2str(results(i).obj_values)]);
%end

    % solutions
segment_lengths = zeros(numel(results) - 1, 1);
for i = 1:numel(results) - 1
    segment_lengths(i) = euclidean_distance(results(i).obj_values, results(i + 1).obj_values);
end

    % Compute the lengths of the segments between all 
    % the neighboring solutions. Delete nearly overlapping solutions.

    % Prescribed distance epsilon calculated based on the average objective values of the results
average_obj_values = mean([results.obj_values]); 

epsilon = 1e-3 * mean(average_obj_values);

    % Compute distances between neighboring solutions and remove nearly overlapping solutions
distances = zeros(ninitial - 1, 1);
indices_to_delete = [];
for i = 1:ninitial - 1
    distances(i) = euclidean_distance(results(i).obj_values, results(i + 1).obj_values);
    if distances(i) < epsilon
        indices_to_delete = [indices_to_delete, i + 1]; 
    end
end

    % Delete nearly overlapping solutions
results(indices_to_delete) = [];

    % Display the updated results
   % Check if there is more than one result before displaying
%if numel(results) > 1
    %disp('Results after removing nearly overlapping solutions:');
    %disp('Lambda | Objective Values');
    %disp('-------|-----------------');
    %for i = 1:numel(results)
    %    disp([num2str(results(i).lambda) '    | ' num2str(results(i).obj_values)]);
    %end

%end
    % Step 5
if numel(results) == 1
    total_t = cputime-ts;
    disp('Adaptive weights method completed2:');
    disp(['Total iterations: ' num2str(countit)]);
    disp('Objective Values:');
    fprintf('%14.6f   %14.6f\n', results.obj_values);
% Display elapsed time
    fprintf('Total runtime: %.2f seconds\n', total_t);
    fprintf('Gurobi runtime: %.2f seconds\n', gurobi_t);    
    return;
end

    % Step 4
    % Determine the number of further refinements
    % in each of the regions

    % Calculate the average length of all segments
average_length = mean(segment_lengths);

    % Set the constant C 
C = 1.1;

    % Determine the number of further refinements for each segment
num_refinements = round(C * segment_lengths / average_length);

    % Display the number of further refinements for each segment
    %disp('Number of Further Refinements:');
    %disp('Segment | Num Refinements');
    %disp('--------|-----------------');
    %for i = 1:numel(num_refinements)
    %    disp([num2str(i) '       | ' num2str(num_refinements(i))]);
    %end

    % Determine the offset distances and compute final Pareto front
nsegments = numel(results) - 1;
offset_distances = zeros(nsegments, 2);

    % Step 6
    % scaling factor for delta_J
scaling_factor = 0.79; % for scaling_factor = 0.5 Total iterations: 9

for i = 1:nsegments
        % Calculate angle θ
    P1 = results(i).obj_values;
    P2 = results(i + 1).obj_values;
    theta = atan(-(P1(2) - P2(2)) / (P1(1) - P2(1)));

        % Determine offset distances with scaling factor
    delta_J = scaling_factor * max(distances);
    delta_1 = delta_J * cos(theta);
    delta_2 = delta_J * sin(theta);

        % Store offset distances
    offset_distances(i, :) = [delta_1, delta_2];
end
    
%disp('Offset Distances:');
%disp('Segment | Delta_J1 | Delta_J2');
%disp('--------|----------|----------');
%for i = 1:nsegments
%    disp([num2str(i) '       | ' num2str(offset_distances(i, 1)) '        | ' num2str(offset_distances(i, 2))]);
%end

    % Step 7: Impose additional inequality constraints and
    % conduct sub-optimization with the weighted-sum method
    % in each of the feasible regions
    
    % submodel with linearized additional constraints
A_ext = [A, zeros(size(A, 1),4*n); ...                                               % original constraints
         lin', zeros(1, 2*n), ones(1, n), zeros(1, n); ...                       % J1 <= P1 - delta1
         lin2', zeros(1, 2*n), zeros(1, n), ones(1, n); ...                      % J2 <= P2 - delta2
         zeros(n, n), 2 * diag(y1_L), zeros(n, n), -eye(n), zeros(n, n); ...         % underestimator 1 obj 1 
         zeros(n, n), 2 * diag(y1_U), zeros(n, n), -eye(n), zeros(n, n); ...         % underestimator 2 obj 1
         zeros(n, n), diag(-(y1_U+y1_L)), zeros(n, n), eye(n), zeros(n, n); ...      % overestimator  1 obj 1 
         zeros(n, n), zeros(n,n), 2 * diag(y2_L), zeros(n, n), -eye(n); ...          % underestimator 1 obj 2
         zeros(n, n), zeros(n,n), 2 * diag(y2_U), zeros(n, n), -eye(n); ...          % underestimator 2 obj 2
         zeros(n, n), zeros(n,n), diag(-(y2_U+y2_L)), zeros(n, n), eye(n); ...       % overestimator  1 obj 2 
         L1', -eye(n), zeros(n,3*n); ...                                             % L1'*x - y1 = 0
         L2', zeros(n,n), -eye(n), zeros(n,2*n) ...                                  % L2'*x - y2 = 0
         ];

b_ext = [b; ...
         P1(1) - delta_1; ...
         P2(2) - delta_2; ...
         y1_L .* y1_L; ...   
         y1_U .* y1_U; ...   
         -(y1_U .* y1_L); ... 
         y2_L .* y2_L; ...   
         y2_U .* y2_U; ...   
         -(y2_U .* y2_L); ... 
         zeros(2*n,1) ...   
         ];

    % Initialize all constraints as '<'
    % Set the sense for the last 2*n constraints as '=' 
sense = repmat('<', size(A_ext, 1), 1);  
sense(end-2*n+1:end) = '=';

clear submodel;

    %Construct Gurobi model
submodel.modelsense = 'min';  
submodel.A = sparse(A_ext);  % Convert A matrix to sparse format
clear A_ext;
submodel.rhs = b_ext;        % Right-hand side of the constraints
submodel.vtype = [repmat('C', nx, 1); repmat('B', mbin, 1); repmat('C', 4*n, 1)];  % nx continuous, mbin binary variables
submodel.sense = sense;
submodel.lb = [x_L; -inf*ones(4*n,1)]; 
submodel.ub = [x_U; inf*ones(4*n,1)];

Q1_ext = zeros(5*n, 5*n);  % Create an extended matrix filled with zeros
Q1_ext(1:n, 1:n) = Q1; 
Q2_ext = zeros(5*n, 5*n);  % Create an extended matrix filled with zeros
Q2_ext(1:n, 1:n) = Q2; 

submodel.multiobj(1).objn = [lin; zeros(4*n,1)];  
submodel.multiobj(1).Q = Q1_ext; 

submodel.multiobj(2).objn = [lin2; zeros(4*n,1)];
submodel.multiobj(2).Q = Q2_ext;
  
    % number refinements for segments that require further refinements
k = num_refinements((num_refinements ~= 0));
    
    % Break out of the loop if there are no segments requiring refinement
%if isempty(k)
%    total_t = cputime-ts;
%    disp('Adaptive weights method completed (k):');
%    disp(['Total iterations: ' num2str(iteration)]);
%    disp('Objective Values:');
%    fprintf('%14.6f   %14.6f\n', results.obj_values);
% Display elapsed time
%    fprintf('Total runtime: %.2f seconds\n', total_t);
%    fprintf('Gurobi runtime: %.2f seconds\n', gurobi_t);    
%    return;
%end

for i = 1:k
        % Define lambda
    lambda = (i - 1) / (k - 1);

        % Set weights in the Gurobi model
    submodel.multiobj(1).weight = lambda;
    submodel.multiobj(2).weight = 1 - lambda;

        % Optimize
    st = cputime;    
    result = gurobi(submodel, params);
    et = cputime-st;
    gurobi_t = et+gurobi_t;

        % Store results
    results(i).lambda = lambda;
    results(i).obj_values = result.objval(:)';
    results(i).x_values = result.x;
end
countit = countit + 1; % count iteration 

segment_lengths = zeros(numel(results) - 1, 1);
for i = 1:numel(results) - 1
    segment_lengths(i) = euclidean_distance(results(i).obj_values, results(i + 1).obj_values);
end

    % Compute the lengths of the segments between all 
    % the neighboring solutions. Delete nearly overlapping solutions.

    % Prescribed distance epsilon calculated based on the average objective values of the results
average_obj_values = mean([results.obj_values]); 

epsilon = 1e-3 * mean(average_obj_values);

    % Compute distances between neighboring solutions and remove nearly overlapping solutions
distances = zeros(ninitial - 1, 1);
indices_to_delete = [];
for i = 1:ninitial - 1
    distances(i) = euclidean_distance(results(i).obj_values, results(i + 1).obj_values);
    if distances(i) < epsilon
        indices_to_delete = [indices_to_delete, i + 1]; 
    end
end

    % Delete nearly overlapping solutions
results(indices_to_delete) = [];

    % Display the updated results
   % Check if there is more than one result before displaying
%if numel(results) > 1
%    disp('Results after removing nearly overlapping solutions:');
%    disp('Lambda | Objective Values');
%    disp('-------|-----------------');
%    for i = 1:numel(results)
%        disp([num2str(results(i).lambda) '    | ' num2str(results(i).obj_values)]);
%    end

%end
    % Step 5
if numel(results) == 1
    total_t = cputime-ts;
    disp('Adaptive weights method completed3:');
    disp(['Total iterations: ' num2str(countit)]);
    disp('Objective Values:');
    fprintf('%14.6f   %14.6f\n', results.obj_values);
    fprintf('Total runtime: %.2f seconds\n', total_t);
    fprintf('Gurobi runtime: %.2f seconds\n', gurobi_t);    
    return;
end
