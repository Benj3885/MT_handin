#python sample_from_map.py --vqvae out/size512_defects_all/vqvae_161.pt  --bottom out/pixelsnail512_defects_all/pixelsnail_bottom_042.pt --batch 16 --size 512
#python sample_from_map.py --vqvae out/size256_chips/vqvae_081.pt  --bottom out/pixelsnail256_defect/pixelsnail_bottom_032.pt --top out/pixelsnail/pixelsnail_top_007.pt --batch 8 --size 256
#python sample_from_map.py --vqvae out/size256_visible/vqvae_021.pt  --bottom out/pixelsnail_visible_transfer_cont/pixelsnail_bottom_020.pt --top out/pixelsnail/pixelsnail_top_007.pt --batch 8 --size 256
#python sample_from_map.py --vqvae out/size256/vqvae_080.pt  --bottom out/pixelsnail256/pixelsnail_bottom_005.pt --top out/pixelsnail/pixelsnail_top_007.pt --batch 8 --size 256
python sample_from_map.py --vqvae out/size512_overfit/vqvae_258.pt  --bottom out/pixelsnail512_overfit/pixelsnail_bottom_066.pt --top out/pixelsnail512/pixelsnail_top_003.pt --batch 1 --size 512
#python sample_from_map.py --vqvae out/vqvae_038.pt  --bottom pixelsnail_top_005.pt --top out/pixelsnail512/pixelsnail_top_003.pt --batch 1 --size 256
#python sample_from_map.py --vqvae out/size512/vqvae_011.pt  --bottom out/pixelsnail512_overfit/pixelsnail_bottom_066.pt --top out/pixelsnail512/pixelsnail_top_003.pt --size 512

