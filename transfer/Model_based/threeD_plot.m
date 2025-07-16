% Replace 'amber_joint_log.csv' with your actual path/filename:

% === User parameters ===
one_cycle    = true;        % set to true to plot only one cycle
cycle_time   = 1.0;         % duration of one cycle in seconds (adjust as needed)

csvFile = 'amber_joint_log3D.csv';
T = readtable(csvFile);

% Time vector
time = T.sim_time;

if one_cycle
    cycle_num  = 5;                  % << which cycle to plot (1-indexed)
    start_t    = (cycle_num-1)*cycle_time;
    end_t      = cycle_num*cycle_time;

    idx = time >= start_t & time <= end_t;

    time           = time(idx) - start_t;   % zero-shift for nicer plots

    % Joint angles
    q1_left_act    = T.q1_left(idx);
    q2_left_act    = T.q2_left(idx);
    q1_right_act   = T.q1_right(idx);
    q2_right_act   = T.q2_right(idx);
    q1_left_req    = T.act_q1_left(idx);
    q2_left_req    = T.act_q2_left(idx);
    q1_right_req   = T.act_q1_right(idx);
    q2_right_req   = T.act_q2_right(idx);

    % COM
    com_x          = T.com_x(idx);
    com_y          = T.com_y(idx);
    com_z          = T.com_z(idx);

    % Foot positions
    cur_l          = [T.cur_foot_l_x(idx), T.cur_foot_l_y(idx), T.cur_foot_l_z(idx)];
    cur_r          = [T.cur_foot_r_x(idx), T.cur_foot_r_y(idx), T.cur_foot_r_z(idx)];
    tgt_l          = [T.tgt_foot_l_x(idx), T.tgt_foot_l_y(idx), T.tgt_foot_l_z(idx)];
    tgt_r          = [T.tgt_foot_r_x(idx), T.tgt_foot_r_y(idx), T.tgt_foot_r_z(idx)];
else
    % Full data
    q1_left_act    = T.q1_left;
    q2_left_act    = T.q2_left;
    q1_right_act   = T.q1_right;
    q2_right_act   = T.q2_right;
    q1_left_req    = T.act_q1_left;
    q2_left_req    = T.act_q2_left;
    q1_right_req   = T.act_q1_right;
    q2_right_req   = T.act_q2_right;
    
    com_x          = T.com_x;
    com_y          = T.com_y;
    com_z          = T.com_z;
    
    cur_l          = [T.cur_foot_l_x, T.cur_foot_l_y, T.cur_foot_l_z];
    cur_r          = [T.cur_foot_r_x, T.cur_foot_r_y, T.cur_foot_r_z];
    tgt_l          = [T.tgt_foot_l_x, T.tgt_foot_l_y, T.tgt_foot_l_z];
    tgt_r          = [T.tgt_foot_r_x, T.tgt_foot_r_y, T.tgt_foot_r_z];
end

% ———————————————————————————————
% 1) Actual vs. commanded joint angles
% ———————————————————————————————
figure;
plot(time, q1_left_act, 'LineWidth',1.5); hold on;
plot(time, q1_left_req, '--', 'LineWidth',1.5); hold off;
xlabel('Time (s)'); ylabel('Angle (rad)');
title('q1\_left: Actual vs. Commanded');
legend('actual','command','Location','best');
grid on;

figure;
plot(time, q2_left_act, 'LineWidth',1.5); hold on;
plot(time, q2_left_req, '--', 'LineWidth',1.5); hold off;
xlabel('Time (s)'); ylabel('Angle (rad)');
title('q2\_left: Actual vs. Commanded');
legend('actual','command','Location','best');
grid on;

figure;
plot(time, q1_right_act, 'LineWidth',1.5); hold on;
plot(time, q1_right_req, '--', 'LineWidth',1.5); hold off;
xlabel('Time (s)'); ylabel('Angle (rad)');
title('q1\_right: Actual vs. Commanded');
legend('actual','command','Location','best');
grid on;

figure;
plot(time, q2_right_act, 'LineWidth',1.5); hold on;
plot(time, q2_right_req, '--', 'LineWidth',1.5); hold off;
xlabel('Time (s)'); ylabel('Angle (rad)');
title('q2\_right: Actual vs. Commanded');
legend('actual','command','Location','best');
grid on;

% ———————————————————————————————
% 2) COM Trajectory (x,y,z)
% ———————————————————————————————
figure;
plot(time, com_x, 'LineWidth',1.5); hold on;
plot(time, com_y, 'LineWidth',1.5);
plot(time, com_z, 'LineWidth',1.5); hold off;
xlabel('Time (s)'); ylabel('CoM Position (m)');
title('Center of Mass Trajectory');
legend('COM\_x','COM\_y','COM\_z','Location','best');
grid on;

% ———————————————————————————————
% 3) 3D Foot Trajectories
% ———————————————————————————————
figure; hold on;
% left current: light green
plot3(cur_l(:,1), cur_l(:,2), cur_l(:,3), 'LineWidth',1.5, 'Color',[0.3,0.8,0.3]);
% left target: dark green
plot3(tgt_l(:,1), tgt_l(:,2), tgt_l(:,3), '--', 'LineWidth',1.5, 'Color',[0,0.5,0]);
% right current: light blue
plot3(cur_r(:,1), cur_r(:,2), cur_r(:,3), 'LineWidth',1.5, 'Color',[0.3,0.3,0.8]);
% right target: dark blue
plot3(tgt_r(:,1), tgt_r(:,2), tgt_r(:,3), '--', 'LineWidth',1.5, 'Color',[0,0,0.5]);
hold off;

xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('Foot Trajectories in 3D');
legend('Left Current','Left Target','Right Current','Right Target','Location','best');
grid on;
view(3);


% ──────────────────────────────────────────────────────────────────────────────
% 3-D  “time–x–z” plot   (one line per foot, current & reference)
% ──────────────────────────────────────────────────────────────────────────────
figure; hold on;
% left current : light green
plot3(time,  cur_l(:,1),  cur_l(:,3),  'LineWidth',1.5, 'Color',[0.4 0.9 0.4]);
% left target  : dark  green
plot3(time,  tgt_l(:,1),  tgt_l(:,3),  '--', 'LineWidth',1.5, 'Color',[0.0 0.5 0.0]);

% right current: light blue
plot3(time,  cur_r(:,1),  cur_r(:,3),  'LineWidth',1.5, 'Color',[0.4 0.4 0.9]);
% right target : dark  blue
plot3(time,  tgt_r(:,1),  tgt_r(:,3),  '--', 'LineWidth',1.5, 'Color',[0.0 0.0 0.5]);

xlabel('Time (s)');
ylabel('Foot X (m)');
zlabel('Foot Z (m)');
title(sprintf('Foot Trajectories (Cycle %d)', cycle_num));
legend('Left Cur','Left Ref','Right Cur','Right ref','Location','best');
grid on;  view(36,24);   % adjust az/el as you like