%% 4-Bus Power Network with Grid-Only Supply + Dynamic RTP + DR Scheduling + MOPSO
% Uses correct 'ActivePower_W_' syntax; Sheet1; scales short trace to 24h; PU conv; hourly means
% Grid supplies all load with DSM; no solar PV; Implements Dynamic RTP, DR scheduling, and MOPSO for cost/emission optimization
clear; clc; close all;

%% Step 1: System Data (per unit, base MVA=100)
nbus = 4;
base_MVA = 100;  % For PU: / (base_MVA * 1e6) to convert W/var to pu

% Bus data: [bus_num, type(1=slack,2=PV,3=PQ), Pd_base approx, Qd_base approx, V_mag(pu), theta(deg)]
bus_data = [
    1, 1, 0.1,  0.06,  1.00, 0;  % Bus 1: Slack + Residential/Domestic + Grid source
    2, 3, 0.2,  0.12,  1.00, 0;  % Bus 2: Water pump
    3, 3, 0.8,  0.48,  1.00, 0;  % Bus 3: Milk processing
    4, 3, 0.5,  0.30,  1.00, 0   % Bus 4: Maize mill
];

% Generator data (grid at Bus 1 only)
gen_data = [
    1, 0.0, 1.00, -0.3, 0.3  % Grid source at Bus 1 (initially 0 pu, adjusted by load balance)
];

%% 2000 data points per file....
%% domestic loads
%% water pump
%% milk processing machine
%% Maize mill machine

filenames = {
    'C:/Users/Baymax/Desktop/Mr.Usamah/Profiles/granular/h_pf/Domestic_loads.xlsx', ...  % Bus 1: Domestic loads
    'C:/Users/Baymax/Desktop/Mr.Usamah/Profiles/granular/h_pf/interpolated_water_pump_hpf_profile.xlsx', ...  % Bus 2: Water pump
    'C:/Users/Baymax/Desktop/Mr.Usamah/Profiles/granular/h_pf/interpolated_milk_processing_plant_hpf_profile.xlsx', ...  % Bus 3: Milk processing
    'C:/Users/Baymax/Desktop/Mr.Usamah/Profiles/granular/h_pf/interpolated_maize_mill_hpf_profile.xlsx'  % Bus 4: Maize mill
};

% Line data
line_data = [
    1, 2, 0.02, 0.10, 0;
    1, 3, 0.01, 0.06, 0;
    1, 4, 0.01, 0.08, 0;
    2, 3, 0.0125, 0.075, 0;
    2, 4, 0.0125, 0.08, 0
];

% Initial billing parameters
B_grid_base = 80;  % Base bid price for Grid ($/MWh)
B_DR_base = 70;    % Base bid for DR ($/MWh)
E_grid = 952.6;    % kg/MWh for Grid (CO2: 950, SO2: 0.5, NOx: 2.1)

% RTP and DR parameters
price_factor_low = 0.7;    % Discount factor for low demand (30% off)
price_factor_high = 1.3;   % Premium factor for high demand (30% more)
demand_threshold = 0.8;    % Threshold for high demand (pu of total load)
DR_max_percentage = 0.2;   % Maximum DR reduction (20% of load)
DR_incentive_factor = 0.5; % Incentive as 50% of base DR price

%% Step 2: Build Ybus
Ybus = zeros(nbus, nbus);
for k = 1:size(line_data, 1)
    i = line_data(k, 1); j = line_data(k, 2);
    r = line_data(k, 3); x = line_data(k, 4); b_shunt = line_data(k, 5);
    z = r + 1j * x; y_series = 1 / z;
    Ybus(i, i) = Ybus(i, i) + y_series + 1j * b_shunt;
    Ybus(j, j) = Ybus(j, j) + y_series + 1j * b_shunt;
    Ybus(i, j) = Ybus(i, j) - y_series;
    Ybus(j, i) = Ybus(j, i) - y_series;
end
G = real(Ybus); B = imag(Ybus);
fprintf('Ybus Built for 4-Bus System with Grid-Only Supply and DSM.\n');

%% Step 3: Read & Scale/Aggregate Load Profiles from Excel
Pd_profiles = zeros(nbus, 24); Qd_profiles = zeros(nbus, 24);
hours = 0:23;
start_date = datetime(2025, 10, 24, 0, 0, 0);

for b = 1:nbus
    filename = filenames{b};
    if ~isfile(filename)
        error('File not found: %s. Please ensure the file exists at the specified path or update the path in filenames{%d}.', filename, b);
    end
    t = readtable(filename);
    disp(['Available columns in ' filename ':']);
    disp(t.Properties.VariableNames);
    if ~ismember('Time', t.Properties.VariableNames) || ...
       ~ismember('ActivePower_W_', t.Properties.VariableNames) || ...
       ~ismember('ReactivePower_var_', t.Properties.VariableNames)
        error('Invalid data format in %s. Expected columns: Time, ActivePower_W_, ReactivePower_var_. Please check your Excel file or adjust column names in the code.', filename);
    end
    time_sec = t.('Time');
    Pd_raw_W = t.('ActivePower_W_');
    Qd_raw_var = t.('ReactivePower_var_');
    max_time = max(time_sec);
    if max_time <= 0
        error('Invalid max_time=%.2f in %s; check Time column contains valid positive values.', max_time, filename);
    end
    scaled_h = double((time_sec / max_time) * 24);
    Pd_pu = Pd_raw_W / (base_MVA * 1e6);
    Qd_pu = Qd_raw_var / (base_MVA * 1e6);
    duration_sec = seconds(scaled_h * 3600);
    time_dt = start_date + duration_sec;
    tt_pd = timetable(time_dt, Pd_pu, 'VariableNames', {'Pd'});
    tt_qd = timetable(time_dt, Qd_pu, 'VariableNames', {'Qd'});
    hourly_pd = retime(tt_pd, 'hourly', 'mean');
    hourly_qd = retime(tt_qd, 'hourly', 'mean');
    Pd_h = hourly_pd.Pd(1:min(24, height(hourly_pd)));
    Qd_h = hourly_qd.Qd(1:min(24, height(hourly_qd)));
    if length(Pd_h) < 24 || length(Qd_h) < 24
        warning('Padding short profile for Bus %d with last value to reach 24 hours.', b);
        Pd_h = [Pd_h; repmat(Pd_h(end), 24 - length(Pd_h), 1)];
        Qd_h = [Qd_h; repmat(Qd_h(end), 24 - length(Qd_h), 1)];
    end
    Pd_profiles(b, :) = Pd_h';
    Qd_profiles(b, :) = Qd_h';
    fprintf('Bus %d (%s): Scaled %d pts (%.1f s → 24h), Hourly P: %.4f to %.4f pu, Q: %.4f to %.4f pu\n', ...
            b, filename, height(t), max_time, min(Pd_profiles(b,:)), max(Pd_profiles(b,:)), ...
            min(Qd_profiles(b,:)), max(Qd_profiles(b,:)));
end

if any(size(Pd_profiles) ~= [nbus, 24]) || any(size(Qd_profiles) ~= [nbus, 24])
    error('Load profile dimensions mismatch. Expected [%d, 24], got [%d, %d] for P and [%d, %d] for Q.', ...
          nbus, size(Pd_profiles,1), size(Pd_profiles,2), size(Qd_profiles,1), size(Qd_profiles,2));
end

%% Step 4: Time-Series Power Flow + Dynamic RTP + DR Scheduling
n_hours = 24;
V_history = zeros(nbus, n_hours); theta_history = zeros(nbus, n_hours);
bus_type = bus_data(:,2);
pq_idx = find(bus_type == 3);  % [2 3 4]
pv_idx = [];                   % No PV buses
n_theta = nbus - 1;            % 3
n_pq = length(pq_idx);         % 3
n_eq = n_theta + n_pq;         % 6
tol = 1e-6; max_iter = 100; damping = 0.3;  % Adjusted for better convergence
Pg_base = zeros(nbus,1); 
for i = 1:size(gen_data, 1)
    Pg_base(gen_data(i,1)) = gen_data(i,2);
end
V_spec_vals = [1.0; 1.0; 1.0; 1.0];  % All buses at 1.0 pu (no PV)

grid_gen_pu = zeros(1, n_hours);
total_cost = zeros(1, n_hours);
total_emission = zeros(1, n_hours);
B_grid_dynamic = zeros(1, n_hours);  % Dynamic grid price
B_DR_dynamic = zeros(1, n_hours);    % Dynamic DR price
DR_reduction_pu = zeros(nbus, n_hours);  % DR reduction per bus

% Initialize table for logging
log_table = table('Size', [n_hours 3], ...
    'VariableTypes', {'double', 'double', 'double'}, ...
    'VariableNames', {'Hour', 'TotalCost', 'TotalEmission'});

% Verify Pd_profiles dimensions
if size(Pd_profiles, 1) ~= nbus || size(Pd_profiles, 2) ~= n_hours
    error('Pd_profiles has incorrect dimensions. Expected [%d, %d], got [%d, %d].', nbus, n_hours, size(Pd_profiles, 1), size(Pd_profiles, 2));
end

for t = 1:n_hours
    Pd = Pd_profiles(:, t); Qd = Qd_profiles(:, t);
    total_load_pu = sum(Pd);
    
    % Dynamic RTP Pricing
    normalized_load = total_load_pu / max(sum(Pd_profiles));  % Normalize load (0 to 1)
    if normalized_load < demand_threshold
        B_grid_dynamic(t) = B_grid_base * price_factor_low;
        B_DR_dynamic(t) = B_DR_base * price_factor_low;
    else
        B_grid_dynamic(t) = B_grid_base * price_factor_high;
        B_DR_dynamic(t) = B_DR_base * price_factor_high;
    end
    
    % DR Scheduling: Trigger DR during high demand
    if normalized_load >= demand_threshold
        for b = 1:nbus
            DR_reduction_pu(b, t) = min(DR_max_percentage * Pd(b), Pd(b) * 0.5);  % Fixed indexing to Pd(b)
            Pd(b) = Pd(b) - DR_reduction_pu(b, t);  % Reduce load with DR
        end
    end
    total_load_pu = sum(Pd);  % Recalculate after DR
    grid_gen_pu(t) = total_load_pu;  % Grid covers all load (no solar)
    Pg_t = Pg_base;
    Pg_t(1) = grid_gen_pu(t);  % Grid generation at Bus 1
    P_spec = Pg_t - Pd; Q_spec = -Qd;
    
    if t == 1
        V = V_spec_vals; theta = zeros(nbus,1);
    else
        V = V_history(:, t-1); theta = theta_history(:, t-1);
        if any(isnan(V)) || any(isnan(theta))
            warning('NaN detected in previous solution for hour %d. Resetting to initial guess.', t);
            V = V_spec_vals; theta = zeros(nbus,1);
        end
    end
    
    % Validate matrix dimensions
    if size(V, 1) ~= nbus || size(theta, 1) ~= nbus || size(G, 1) ~= nbus || size(B, 1) ~= nbus
        error('Matrix dimension mismatch. V, theta, G, or B have incorrect sizes. Expected %d x 1 or %d x %d.', nbus, nbus, nbus);
    end
    
    converged = false;
    for iter = 1:max_iter
        P_calc = zeros(nbus,1); Q_calc = zeros(nbus,1);
        for i = 1:nbus
            sumP = 0; sumQ = 0;
            for j = 1:nbus
                delta = theta(i) - theta(j);
                c = cos(delta); s = sin(delta);
                sumP = sumP + V(j) * (G(i,j) * c + B(i,j) * s);
                sumQ = sumQ + V(j) * (G(i,j) * s - B(i,j) * c);
            end
            P_calc(i) = V(i) * sumP;
            Q_calc(i) = V(i) * sumQ;
        end
        
        deltaP = P_spec - P_calc; deltaP(1) = 0;
        deltaQ = Q_spec - Q_calc;
        deltaX = [deltaP(2:end); deltaQ(pq_idx)];
        
        if max(abs(deltaX)) < tol
            converged = true; break;
        end
        
        J = zeros(n_eq, n_eq);
        for r = 1:n_theta
            i = r + 1;
            for c = 1:n_theta
                j = c + 1;
                if i == j
                    sum_off = 0;
                    for k = 1:nbus
                        if k ~= i
                            dk = theta(i) - theta(k);
                            sum_off = sum_off + V(i) * V(k) * (G(i,k) * sin(dk) - B(i,k) * cos(dk));
                        end
                    end
                    J(r, c) = -sum_off;
                else
                    delta_ij = theta(i) - theta(j);
                    s_ij = sin(delta_ij); c_ij = cos(delta_ij);
                    J(r, c) = V(i) * V(j) * (G(i,j) * s_ij - B(i,j) * c_ij);
                end
            end
        end
        
        for r = 1:n_theta
            i = r + 1;
            for cc = 1:n_pq
                m = pq_idx(cc);
                delta_im = theta(i) - theta(m);
                c_im = cos(delta_im); s_im = sin(delta_im);
                J(r, n_theta + cc) = V(i) * (G(i,m) * c_im + B(i,m) * s_im);
            end
        end
        
        for rr = 1:n_pq
            i = pq_idx(rr);
            for c = 1:n_theta
                j = c + 1;
                delta_ij = theta(i) - theta(j);
                s_ij = sin(delta_ij); c_ij = cos(delta_ij);
                if i == j
                    J(n_theta + rr, c) = P_calc(i) - V(i)^2 * G(i,i);
                else
                    J(n_theta + rr, c) = -V(i) * V(j) * (G(i,j) * c_ij + B(i,j) * s_ij);
                end
            end
        end
        
        for rr = 1:n_pq
            i = pq_idx(rr);
            for cc = 1:n_pq
                m = pq_idx(cc);
                if i == m
                    J(n_theta + rr, n_theta + cc) = Q_calc(i) / V(i) - V(i) * B(i,i);
                else
                    delta_im = theta(i) - theta(m);
                    s_im = sin(delta_im); c_im = cos(delta_im);
                    J(n_theta + rr, n_theta + cc) = V(i) * (G(i,m) * s_im - B(i,m) * c_im);
                end
            end
        end
        
        dX = (J \ deltaX) * damping;
        theta(2:end) = theta(2:end) + dX(1:n_theta);
        V(pq_idx) = V(pq_idx) + dX(n_theta+1:end);
        V(1) = V_spec_vals(1);  % Slack bus voltage fixed
    end
    
    V_history(:, t) = V; theta_history(:, t) = theta;
    if ~converged
        error('Power flow for hour %d did not converge.', t);
    end
    
    % Billing calculation with dynamic RTP and DR
    P_grid_MWh = grid_gen_pu(t) * (base_MVA / 1000) * 1;  % 1 hour interval
    P_DR_MWh = sum(DR_reduction_pu(:, t)) * (base_MVA / 1000) * 1;  % DR reduction in MWh
    cost_grid = P_grid_MWh * B_grid_dynamic(t);  % Dynamic grid cost
    cost_DSM = P_DR_MWh * B_DR_dynamic(t) - (P_DR_MWh * B_DR_dynamic(t) * DR_incentive_factor);  % Cost minus incentive
    total_cost(t) = cost_grid + cost_DSM;  % No solar or storage costs
    
    emission_grid = P_grid_MWh * E_grid;  % 952.6 kg/MWh for grid (CO2, SO2, NOx)
    total_emission(t) = emission_grid;  % No solar or storage emissions
    
    % Apply scaling factor of 10,000 to cost and emission
    scaled_cost = total_cost(t) * 10000;
    scaled_emission = total_emission(t) * 10000;
    
    % Log data to table
    log_table.Hour(t) = t;
    log_table.TotalCost(t) = scaled_cost;
    log_table.TotalEmission(t) = scaled_emission;
end

grid_gen_kW = grid_gen_pu * base_MVA * 1000;

%% Step 5: MOPSO Implementation
n_particles = 200;  % Number of particles
n_iterations = 300;  % Number of iterations
dim = n_hours * 2;  % Dimensions: grid_gen_pu and DR_reduction_pu for each hour

% Initialize particle positions and velocities
particles = zeros(n_particles, dim);
velocities = zeros(n_particles, dim);
p_best = zeros(n_particles, dim);
p_best_cost = inf(n_particles, 1);  % Initial best cost for each particle
p_best_emission = inf(n_particles, 1);  % Initial best emission for each particle
global_best_cost_history = inf(1, n_iterations);  % Best cost across all particles
global_best_emission_history = inf(1, n_iterations);  % Best emission across all particles
repository = [];  % Pareto front repository
p_best_idx = zeros(n_particles, 1);  % Index to track which particle's p_best corresponds to repository
w_max = 0.9;  % Initial inertia weight
w_min = 0.4;  % Final inertia weight
c1 = 2.0;     % Cognitive coefficient
c2 = 2.0;     % Social coefficient
v_max = 0.2;  % Maximum velocity

% Random initialization within constraints using base case data
for i = 1:n_particles
    for t = 1:n_hours
        total_load = sum(Pd_profiles(:, t));
        max_grid = total_load;  % Grid can supply up to total load
        max_dr = DR_max_percentage * total_load;
        particles(i, (t-1)*2+1) = grid_gen_pu(t) * (0.8 + 0.4 * rand);  % Perturb around base grid
        particles(i, (t-1)*2+2) = DR_reduction_pu(1, t) * (0.8 + 0.4 * rand);  % Perturb around base DR
        particles(i, (t-1)*2+1) = max(0, min(max_grid, particles(i, (t-1)*2+1)));
        particles(i, (t-1)*2+2) = max(0, min(max_dr, particles(i, (t-1)*2+2)));
    end
end

% MOPSO main loop
for iter = 1:n_iterations
    w = w_max - (w_max - w_min) * (iter - 1) / (n_iterations - 1);  % Dynamically adjust inertia weight
    
    for i = 1:n_particles
        % Evaluate particle
        particle = particles(i, :);
        grid_gen_pu_opt = particle(1:2:end);
        dr_reduction_pu_opt = zeros(nbus, n_hours);
        for t = 1:n_hours
            dr_reduction_pu_opt(1, t) = particle((t-1)*2+2);  % Use first bus DR as representative
            total_load_t = sum(Pd_profiles(:, t));
            dr_reduction_t = min(dr_reduction_pu_opt(1, t), DR_max_percentage * total_load_t);
            for b = 1:nbus
                dr_reduction_pu_opt(b, t) = dr_reduction_t * (Pd_profiles(b, t) / total_load_t);  % Distribute DR proportionally
            end
            grid_gen_pu_t = min(grid_gen_pu_opt(t), total_load_t - sum(dr_reduction_pu_opt(:, t)));  % Grid covers remaining load
            if grid_gen_pu_t < 0, grid_gen_pu_t = 0; end
        end
        
        total_cost_i = 0;
        total_emission_i = 0;
        
        for t = 1:n_hours
            P_grid_MWh = grid_gen_pu_opt(t) * (base_MVA / 1000) * 1;
            P_DR_MWh = sum(dr_reduction_pu_opt(:, t)) * (base_MVA / 1000) * 1;
            cost_grid = P_grid_MWh * B_grid_dynamic(t);
            cost_dsm = P_DR_MWh * B_DR_dynamic(t) - (P_DR_MWh * B_DR_dynamic(t) * DR_incentive_factor);
            emission_grid = P_grid_MWh * E_grid;
            
            total_cost_i = total_cost_i + (cost_grid + cost_dsm);
            total_emission_i = total_emission_i + emission_grid;
        end
        
        % Update p_best
        if total_cost_i < p_best_cost(i) || total_emission_i < p_best_emission(i)
            if total_cost_i < p_best_cost(i) && total_emission_i <= p_best_emission(i)
                p_best(i, :) = particle;
                p_best_cost(i) = total_cost_i;
                p_best_emission(i) = total_emission_i;
                p_best_idx(i) = i;
            elseif total_emission_i < p_best_emission(i) && total_cost_i <= p_best_cost(i)
                p_best(i, :) = particle;
                p_best_cost(i) = total_cost_i;
                p_best_emission(i) = total_emission_i;
                p_best_idx(i) = i;
            end
        end
        
        % Update repository (dominance check)
        new_solutions = [total_cost_i, total_emission_i];
        if isempty(repository)
            repository = [repository; new_solutions];
        else
            dominated = false;
            for j = 1:size(repository, 1)
                if all(repository(j, :) <= new_solutions) && any(repository(j, :) < new_solutions)
                    dominated = true; break;
                end
            end
            if ~dominated
                repository(end+1, :) = new_solutions;
                to_keep = true(size(repository, 1), 1);
                for j = 1:size(repository, 1)
                    for k = 1:size(repository, 1)
                        if j ~= k && all(repository(k, :) <= repository(j, :)) && any(repository(k, :) < repository(j, :))
                            to_keep(j) = false; break;
                        end
                    end
                end
                repository = repository(to_keep, :);
            end
        end
    end
    
    % Update global best history
    [global_best_cost_history(iter), min_cost_idx] = min(p_best_cost);
    [global_best_emission_history(iter), min_emission_idx] = min(p_best_emission);
end

% Scale MOPSO results by 10,000 for consistency
repository = repository * 10000;

%% Step 6: Results Table
fprintf('\nOptimal Solutions:\n');
optimal_table = array2table(repository, 'VariableNames', {'TotalCost', 'TotalEmission'});
disp(optimal_table);

%% Step 7: Plots
figure('Position', [100, 100, 1200, 800]);
for b = 1:nbus
    subplot(2,2,b);
    yyaxis left; plot(hours, Pd_profiles(b,:), 'b-', 'LineWidth', 2); ylabel('Pd (pu)');
    yyaxis right; plot(hours, Qd_profiles(b,:), 'r--', 'LineWidth', 2); ylabel('Qd (pu)');
    title(sprintf('Bus %d Load Profile (Scaled Hourly, pu)', b));
    xlabel('Hour of Day'); grid on;
end
sgtitle('24-Hour Real Load Profiles: Scaled from Excel Transients');

figure('Position', [100, 100, 1200, 800]);
subplot(2,2,1);
plot(hours, V_history', 'LineWidth', 1.5); 
legend(arrayfun(@(x) sprintf('Bus %d',x), 1:nbus, 'UniformOutput', false));
xlabel('Hour'); ylabel('V (pu)'); title('Voltage levels'); grid on;
legend('Location','best');

subplot(2,2,2);
plot(hours, grid_gen_kW, 'b--', 'LineWidth', 2, 'DisplayName', 'Grid Generation (kW)');
xlabel('Hour of Day'); ylabel('Power (kW)'); title('Grid Generation Over 24 Hours (Grid + DSM)');
grid on;
legend('Location', 'best');

subplot(2,2,3);
net_P = sum(Pd_profiles, 1) - grid_gen_pu - sum(DR_reduction_pu);  % Adjusted for DR
net_Q = sum(Qd_profiles, 1);  % Q unchanged
yyaxis left; plot(hours, net_P, 'b-', 'LineWidth', 2); ylabel('Net Active Power (pu)');
yyaxis right; plot(hours, net_Q, 'r--', 'LineWidth', 2); ylabel('Total Reactive Power (pu)');
xlabel('Hour of Day'); title('Total System Net Power Consumption Over 24 Hours (Grid + DSM)');
grid on;
legend('Net P (Active, Load - Grid - DR)', 'Total Q (Reactive)', 'Location', 'best');

subplot(2,2,4)
yyaxis left; plot(hours, total_cost, 'k-', 'LineWidth', 2); ylabel('Operational Cost ($)');
yyaxis right; plot(hours, total_emission, 'c--', 'LineWidth', 2); ylabel('Pollution Emissions (kg)');
xlabel('Hour of Day'); title('Objective Functions: Cost and Emissions Over 24 Hours (Base Case)');
grid on;
legend('Operational Cost', 'Pollution Emissions', 'Location', 'best');
sgtitle("");


figure('Position', [100, 100, 800, 600]);
subplot(1,2,1);
plot(hours, B_grid_dynamic, 'b-', 'LineWidth', 2, 'DisplayName', 'Grid Price ($/MWh)');
hold on;
plot(hours, B_DR_dynamic, 'r-.', 'LineWidth', 2, 'DisplayName', 'DR Price ($/MWh)');
xlabel('Hour of Day'); ylabel('Price ($/MWh)'); title('Dynamic Real-Time Pricing Over 24 Hours');
grid on;
legend('Location', 'best');
hold off;

subplot(1,2,2)
for b = 1:nbus
    plot(hours, DR_reduction_pu(b,:), 'LineWidth', 1.5, 'DisplayName', sprintf('Bus %d DR Reduction (pu)', b));
    hold on;
end
xlabel('Hour of Day'); ylabel('DR Reduction (pu)'); title('Demand Response Reduction Per Bus Over 24 Hours');
grid on;
legend('Location', 'best');
hold off;

sgtitle("")

% MOPSO Plots
figure('Position', [100, 100, 800, 600]);
plot(p_best_cost * 10000, p_best_emission * 10000, 'b.', 'LineWidth', 2, 'MarkerSize', 10);  % All p_best points
hold on;
ideal_point = [min(p_best_cost), min(p_best_emission)] * 10000;
distances = sqrt(sum((repository - repmat(ideal_point, size(repository, 1), 1)).^2, 2));
[~, best_idx] = min(distances);
best_point = repository(best_idx, :);
best_particle_idx = p_best_idx(find(p_best_cost == best_point(1)/10000, 1));
plot(best_point(1), best_point(2), 'r*', 'LineWidth', 2, 'MarkerSize', 15, 'DisplayName', 'Best Point (Knee)');
xlabel('Operational Cost ($)');
ylabel('Pollution Emissions (kg)');
title('Pareto Front: Cost vs. Emissions with Best Point (RTP)');
grid on;
legend('All Non-Dominated Points', 'Best Point', 'Location', 'best');
hold off;

%% space for convergence plots if needed...

%% Step 9: Optimal Scheduling Analysis
if ~isempty(best_particle_idx)
    optimal_particle = p_best(best_particle_idx, :);
    optimal_grid_gen_pu = optimal_particle(1:2:end);
    optimal_dr_reduction_pu = zeros(nbus, n_hours);
    for t = 1:n_hours
        optimal_dr_reduction_pu(1, t) = optimal_particle((t-1)*2+2);  % Use first bus DR as representative
        total_load_t = sum(Pd_profiles(:, t));
        dr_reduction_t = min(optimal_dr_reduction_pu(1, t), DR_max_percentage * total_load_t);
        for b = 1:nbus
            optimal_dr_reduction_pu(b, t) = dr_reduction_t * (Pd_profiles(b, t) / total_load_t);  % Distribute DR proportionally
        end
    end
    
    optimal_grid_gen_kW = optimal_grid_gen_pu * base_MVA * 1000;
    optimal_dr_reduction_kW = optimal_dr_reduction_pu * base_MVA * 1000;
    
    scheduling_table = table(hours', optimal_grid_gen_kW', sum(optimal_dr_reduction_kW, 1)', ...
        'VariableNames', {'Hour', 'Grid_Gen_kW', 'DR_Reduction_kW'});
    fprintf('\nOptimal Scheduling for Grid and DR (Best Point, kW):\n');
    disp(scheduling_table);
    
    figure('Position', [100, 100, 800, 600]);
    plot(hours, optimal_grid_gen_kW, 'b--', 'LineWidth', 2, 'DisplayName', 'Grid Generation (kW)');
    hold on;
    plot(hours, sum(optimal_dr_reduction_kW, 1), 'r-.', 'LineWidth', 2, 'DisplayName', 'DR Reduction (kW)');
    xlabel('Hour of Day');
    ylabel('Power (kW)');
    title('Optimal Scheduling: Grid and DR Over 24 Hours');
    grid on;
    legend('Location', 'best');
    hold off;
end

%% Step 8: Sample Results and Log Table
fprintf('\nSample PF at Hour 12 (Midday):\n');
fprintf('Bus | V (pu) | theta (deg) | P_load (pu) | Grid (pu) | DR (pu) | Net P (pu)\n');
for i = 1:nbus
    grid_i = (i == 1) * grid_gen_pu(12);
    dr_i = DR_reduction_pu(i, 12);
    net_p_i = Pd_profiles(i,12) - grid_i - dr_i + P_calc(i);
    fprintf('%d   | %.3f  | %.3f     | %.4f      | %.4f      | %.4f    | %.4f\n', ...
            i, V_history(i,12), theta_history(i,12)*180/pi, Pd_profiles(i,12), grid_i, dr_i, net_p_i);
end
fprintf('Total Net Load at Hour 12: P=%.4f pu, Q=%.4f pu\n', sum(net_P(12)), sum(net_Q(12)));
fprintf('Total Grid Generation at Hour 12: %.1f kW (%.4f pu), Price=%.2f $/MWh, Cost=%.2f $, Emission=%.2f kg\n', ...
    grid_gen_kW(12), grid_gen_pu(12), B_grid_dynamic(12), P_grid_MWh * B_grid_dynamic(12), P_grid_MWh * E_grid);
fprintf('Total DR Reduction at Hour 12: %.4f pu, Price=%.2f $/MWh, Incentive=%.2f $, Net Cost=%.2f $\n', ...
    sum(DR_reduction_pu(:,12)), B_DR_dynamic(12), P_DR_MWh * B_DR_dynamic(12) * DR_incentive_factor, cost_DSM);
fprintf('Operational Cost at Hour 12: %.2f $\n', total_cost(12));
fprintf('Pollution Emissions at Hour 12: %.2f kg\n', total_emission(12));

% Display logged table
fprintf('\nHourly Log Table\n');
disp(log_table);