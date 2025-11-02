%% 4-Bus Power Network with Grid-Only Supply and No DSM or Solar PV
% Uses correct 'ActivePower_W_' syntax; Sheet1; scales short trace to 24h; PU conv; hourly means
% Grid supplies all load; no solar PV or DSM implemented
clear; clc; close all;

%% Step 1: System Data (per unit, base MVA=100)
nbus = 4;
base_MVA = 100;  % For PU: / (base_MVA * 1e6) to convert W/var to pu

% Bus data: [bus_num, type(1=slack,2=PV,3=PQ), Pd_base approx, Qd_base approx, V_mag(pu), theta(deg)]
bus_data = [
    1, 1, 0.1,  0.06,  1.00, 0;  % Bus 1: Slack + Residential/Domestic + Grid source
    2, 3, 0.2,  0.12,  1.00, 0;  % Bus 2: Water pump (no PV)
    3, 3, 0.8,  0.48,  1.00, 0;  % Bus 3: Milk processing
    4, 3, 0.5,  0.30,  1.00, 0   % Bus 4: Maize mill
];

% Generator data (grid at Bus 1 only)
gen_data = [
    1, 0.0, 1.00, -0.3, 0.3  % Grid source at Bus 1 (initially 0 pu, adjusted by load balance)
];

% File paths for load profiles (solar files not used in this scenario)
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

% Billing parameters (grid-only)
B_grid = 80;  % Fixed grid price ($/MWh)
E_grid = 952.6;  % kg/MWh for Grid (CO2: 950, SO2: 0.5, NOx: 2.1)

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
fprintf('Ybus Built for 4-Bus System with Grid-Only Supply (No DSM or Solar PV).\n');

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

%% Step 4: Time-Series Power Flow with Grid-Only Supply
n_hours = 24;
V_history = zeros(nbus, n_hours); theta_history = zeros(nbus, n_hours);
bus_type = bus_data(:,2);
pq_idx = find(bus_type == 3);  % [2 3 4]
pv_idx = find(bus_type == 2);  % []
n_theta = nbus - 1;  % 3
n_pq = length(pq_idx);  % 3
n_eq = n_theta + n_pq;  % 6
tol = 1e-6; max_iter = 100; damping = 0.3;  % Adjusted for better convergence
Pg_base = zeros(nbus,1); 
for i = 1:size(gen_data, 1)
    Pg_base(gen_data(i,1)) = gen_data(i,2);
end
V_spec_vals = [1.0; 1.0; 1.0; 1.0];  % All buses at 1.0 pu voltage (no PV)

grid_gen_pu = zeros(1, n_hours);
total_cost = zeros(1, n_hours);
total_emission = zeros(1, n_hours);

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
    grid_gen_pu(t) = total_load_pu;  % Grid supplies all load (no solar or DR)
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
    
    % Billing calculation with grid-only supply
    P_grid_MWh = grid_gen_pu(t) * (base_MVA / 1000) * 1;  % 1 hour interval
    cost_grid = P_grid_MWh * B_grid;  % Grid cost
    total_cost(t) = cost_grid;
    emission_grid = P_grid_MWh * E_grid;  % Grid emission
    total_emission(t) = emission_grid;
    
    % Apply scaling factor of 10,000 to cost and emission
    scaled_cost = total_cost(t) * 10000;
    scaled_emission = total_emission(t) * 10000;
    
    % Log data to table
    log_table.Hour(t) = t;
    log_table.TotalCost(t) = scaled_cost;
    log_table.TotalEmission(t) = scaled_emission;
end

grid_gen_kW = grid_gen_pu * base_MVA * 1000;

%% Step 5: Plots
% Load Profiles
figure('Position', [100, 100, 1200, 800]);
for b = 1:nbus
    subplot(2,2,b);
    yyaxis left; plot(hours, Pd_profiles(b,:), 'b-', 'LineWidth', 2); ylabel('Pd (pu)');
    yyaxis right; plot(hours, Qd_profiles(b,:), 'r--', 'LineWidth', 2); ylabel('Qd (pu)');
    title(sprintf('Bus %d Load Profile (Scaled Hourly, pu)', b));
    xlabel('Hour of Day'); grid on;
end
sgtitle('24-Hour Real Load Profiles');

% Voltage Levels
figure('Position', [100, 100, 1200, 800]);
subplot(2,2,1)
plot(hours, V_history', 'LineWidth', 1.5); 
legend(arrayfun(@(x) sprintf('Bus %d',x), 1:nbus, 'UniformOutput', false));
xlabel('Hour'); ylabel('V (pu)'); title('Voltage levels'); grid on;
legend("Location",'best')

% Grid Generation
subplot(2,2,2);
plot(hours, grid_gen_kW, 'b--', 'LineWidth', 2, 'DisplayName', 'Grid Generation (kW)');
xlabel('Hour of Day'); ylabel('Power (kW)'); title('Grid Generation Over 24 Hours (Grid-Only)');
grid on;
legend('Location', 'best');

% Net Power Consumption
subplot(2,2,3);
net_P = sum(Pd_profiles, 1) - grid_gen_pu;  % No solar or DR
net_Q = sum(Qd_profiles, 1);  % Q unchanged
yyaxis left; plot(hours, net_P, 'b-', 'LineWidth', 2); ylabel('Net Active Power (pu)');
yyaxis right; plot(hours, net_Q, 'r--', 'LineWidth', 2); ylabel('Total Reactive Power (pu)');
xlabel('Hour of Day'); title('Total System Net Power Consumption Over 24 Hours (Grid-Only)');
grid on;
legend('Net P (Active, Load - Grid)', 'Total Q (Reactive)', 'Location', 'best');

% Objective Functions: Cost and Emissions
subplot(2,2,4);
yyaxis left; plot(hours, total_cost, 'k-', 'LineWidth', 2); ylabel('Operational Cost ($)');
yyaxis right; plot(hours, total_emission, 'c--', 'LineWidth', 2); ylabel('Pollution Emissions (kg)');
xlabel('Hour of Day'); title('Objective Functions: Cost and Emissions Over 24 Hours (Grid-Only)');
grid on;
legend('Operational Cost', 'Pollution Emissions', 'Location', 'best');

%% Step 6: Sample Results and Log Table
fprintf('\nSample PF at Hour 12 (Midday):\n');
fprintf('Bus | V (pu) | theta (deg) | P_load (pu) | Grid (pu) | Net P (pu)\n');
for i = 1:nbus
    grid_i = (i == 1) * grid_gen_pu(12);
    net_p_i = Pd_profiles(i,12) - grid_i + P_calc(i);
    fprintf('%d   | %.3f  | %.3f     | %.4f      | %.4f      | %.4f\n', ...
            i, V_history(i,12), theta_history(i,12)*180/pi, Pd_profiles(i,12), grid_i, net_p_i);
end
fprintf('Total Net Load at Hour 12: P=%.4f pu, Q=%.4f pu\n', sum(net_P(12)), sum(net_Q(12)));
fprintf('Total Grid Generation at Hour 12: %.1f kW (%.4f pu), Cost=%.2f $, Emission=%.2f kg\n', ...
    grid_gen_kW(12), grid_gen_pu(12), P_grid_MWh * B_grid, P_grid_MWh * E_grid);
fprintf('Operational Cost at Hour 12: %.2f $\n', total_cost(12));
fprintf('Pollution Emissions at Hour 12: %.2f kg\n', total_emission(12));

% Display logged table
fprintf('\nHourly Log Table:\n');
disp(log_table);