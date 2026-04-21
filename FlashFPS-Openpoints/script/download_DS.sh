mkdir -p data/S3DIS/
cd data/S3DIS
gdown --fuzzy https://drive.google.com/file/d/14FdvE02kMUde4dLWlCH_ZVsHVpgwdBlP/view?usp=share_link
tar -xvf s3disfull.tar
cd ../../

cd data
gdown https://drive.google.com/uc?id=1uWlRPLXocqVbJxPvA2vcdQINaZzXf1z_
tar -xvf ScanNet.tar