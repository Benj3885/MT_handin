#python extract_code.py /Data/Real/CAM3/Good/current_version/train/A --size 512 --ckpt out/size512_defects_all/vqvae_161.pt --name lmdb_datasets/good_vials512
#python extract_code.py /Data/Real/CAM3/Good/current_version/train --size 512 --ckpt out/size512/vqvae_011.pt --name lmdb_datasets/vials512
#python extract_code.py /Data/Real/CAM3/Good/current_version/validation/C --size 256 --ckpt out/size256_report_val/vqvae_008.pt --name lmdb_datasets/vials256_report_val
#python extract_code.py /home/bjeh/BJEH/datasets/classifier_set/train --size 512 --ckpt out/size512_overfit/vqvae_258.pt --name lmdb_datasets/size512_overfit_train
python extract_code.py /home/bjeh/BJEH/datasets/classifier_set/val --size 512 --ckpt out/size512_overfit_no_trans/vqvae_500.pt --name lmdb_datasets/size512_overfit_val_no_trans

