path=("03-22-03_smolvla_vanilla_spatial_except_2_4_angle_from_0" "03-22-52_smolvla_basis_concat_spatial_except_2_4_angle_from_0" "03-23-25_smolvla_basis_spatial_except_2_4_angle_from_0" "03-37-11_smolvla_basis_vision_frozen_spatial_except_2_4_angle_from_0")
ROOT="/home/kwonmc/jiyun/lerobot-VAI"
checkpoint_path="$ROOT/outputs/train/2026-01-18"
for i in "${path[@]}"
do
    scp -r kwonmc@61.107.200.23:$checkpoint_path/$i/checkpoints/100000 /data1/local/lerobot-VAI/outputs/train/2026-01-18/$i/checkpoints/
done