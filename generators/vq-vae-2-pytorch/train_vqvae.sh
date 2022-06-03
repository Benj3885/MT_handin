#python train_vqvae.py /Data/Real/CAM3/Good/current_version/train --out size256 --batch 128 --size 256
#python train_vqvae.py /home/bjeh/BJEH/datasets/CAM3/Visible/ChipB --out size256_visible --batch 64 --size 256 --save_interval 10
#python train_vqvae.py /Data/Real/CAM3/Good/current_version/validation/A --out size512_good --batch 128 --size 512
#python train_vqvae.py /Data/Real/CAM3/Good/current_version/train --out size256_report_val --batch 256 --size 256
python train_vqvae.py /home/bjeh/BJEH/datasets/classifier_set/train --out size512_overfit_no_trans --batch 64 --size 512 --val_path /home/bjeh/BJEH/datasets/classifier_set/val
#python train_vqvae.py /Data/Real/CAM3/Good/current_version/train --out size512 --batch 64 --size 512 --ckpt out/size512/vqvae_011.pt --off_set 11
#/Data/Real/CAM3/Good/current_version/train