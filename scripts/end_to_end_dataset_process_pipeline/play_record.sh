# !/usr/bin/env bash
tmux kill-pane -a
tmux kill-session -t apollo-ws
new_session="apollo-ws"
tmux new-session -s $new_session -d
tmux split-window -v
tmux split-window -p 66;
tmux split-window -d
tmux split-window -h
###################### For testing ######################## 
# tmux send-keys -t $new_session:0.1 'echo listening' C-m
# tmux send-keys -t $new_session:0.2 'echo playing' C-m
############################################################
# Launch the module
# Uncommend if want to 
# tmux send-keys -t $session:0.1 "bootstrap.sh; tmux wait-for -S UI" C-m
# tmux wait-for UI
wait
# tmux send-keys -t $session:0.1 "cyber_launch start /apollo/modules/transform/launch/static_transform_nu.launch" C-m;
# # tmux wait-for transform
# tmux send-keys -t $session:0.2 "cyber_launch start /apollo/modules/perception/production/launch/perception_all_nu.launch" C-m;
# sleep 10s

for file in *_detection.record; do
    filename=$(echo $file | cut -d '.' -f 1)
    tmux send-keys -t $session:0.1 "cyber_launch start /apollo/modules/transform/launch/static_transform_nu.launch" C-m;
    tmux send-keys -t $session:0.2 "cyber_launch start /apollo/modules/perception/production/launch/perception_all_nu.launch" C-m;
    sleep 15s
    tmux send-keys -t $session:0.3 "cyber_recorder record -c /apollo/perception/obstacles -o ./$filename.record" C-m;
    tmux send-keys -t $session:0.4 "cyber_recorder play -f $file; tmux wait-for -S play_process" C-m;
    tmux wait-for play_process
    tmux send-keys -t $session:0.3 C-c;
    tmux send-keys -t $session:0.1 C-c;
    tmux send-keys -t $session:0.2 C-c;
    sleep 10s
done
tmux send-keys -t $session:0.1 "echo '----------- processing complete. -----------'" C-m;
tmux send-keys -t $session:0.2 "echo '----------- processing complete. -----------'" C-m;
tmux send-keys -t $session:0.3 "echo '----------- processing complete. -----------'" C-m;
tmux send-keys -t $session:0.4 "echo '----------- processing complete. -----------'" C-m;
tmux -2 attach-session -d