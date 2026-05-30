apt-get update -qq && apt-get install -y -qq aria2 pigz pv tmux htop unzip p7zip-full > /dev/null 2>&1 && echo 'SYS_PACKAGES_OK'
pip install -q --upgrade pip && pip install -q kaggle timm albumentations scikit-learn pandas matplotlib seaborn tqdm wandb safetensors pillow opencv-python-headless && echo 'PIP_PACKAGES_OK'
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
echo 'ENV_SETUP_COMPLETE'
exit
