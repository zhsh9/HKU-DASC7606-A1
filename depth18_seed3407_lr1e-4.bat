@echo off
setlocal enabledelayedexpansion

:: init suffix
set /a "suffix=1"

:: set times
set /a "times=10"

for /l %%i in (1,1,%times%) do (
    python train.py --coco_path ./data --output_path ./model6_r!suffix! --depth 18 --seed 3407 --learning_rate 0.0001 --epochs 20 > log/train_depth18_seed3407_lr1e-4_epochs20_r!suffix!.log
    :: increase suffix
    set /a "suffix+=1"
)

echo Training completed!
endlocal