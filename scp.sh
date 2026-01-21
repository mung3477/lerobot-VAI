# path=("03-22-03_smolvla_vanilla_spatial_except_2_4_angle_from_0" "03-22-52_smolvla_basis_concat_spatial_except_2_4_angle_from_0" "03-23-25_smolvla_basis_spatial_except_2_4_angle_from_0" "03-37-11_smolvla_basis_vision_frozen_spatial_except_2_4_angle_from_0")
path=("22-13-43_smolvla_vanilla_10" "22-33-07_smolvla_vanilla_object" "22-35-43_smolvla_basis_concat_10" "22-37-55_smolvla_basis_concat_object")

ROOT="/home/kwonmc/jiyun/lerobot-VAI"
checkpoint_path="$ROOT/outputs/train/2026-01-20"
for i in "${path[@]}"
do
    mkdir -p /data1/local/lerobot-VAI/outputs/train/2026-01-20/
    mkdir -p /data1/local/lerobot-VAI/outputs/train/2026-01-20/$i/
    mkdir -p /data1/local/lerobot-VAI/outputs/train/2026-01-20/$i/checkpoints/
    scp -r nipa-70:$checkpoint_path/$i/checkpoints/100000 /data1/local/lerobot-VAI/outputs/train/2026-01-20/$i/checkpoints/
    scp -r /data1/local/lerobot-VAI/outputs/train/2026-01-20/$i/checkpoints/100000 honolulu11111:/root/Desktop/workspace/jiyun/lerobot-VAI/outputs/train/2026-01-20/$i/checkpoints/
done

