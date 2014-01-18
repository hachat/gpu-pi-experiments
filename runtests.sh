cd pi-curand
make clean
make sp
../../bin/linux/release/pi-curand 256 2
echo SP===================
../../bin/linux/release/pi-curand 1024 2
echo SP===================
../../bin/linux/release/pi-curand 4096 2
echo SP===================

../../bin/linux/release/pi-curand 256 4
echo SP===================
../../bin/linux/release/pi-curand 1024 4
echo SP===================
../../bin/linux/release/pi-curand 4096 4
echo SP===================

../../bin/linux/release/pi-curand 256 8
echo SP===================
../../bin/linux/release/pi-curand 1024 8
echo SP===================
../../bin/linux/release/pi-curand 4096 8
echo SP===================

make clean

make dp
../../bin/linux/release/pi-curand 256 2
echo DP===================
../../bin/linux/release/pi-curand 1024 2
echo DP===================
../../bin/linux/release/pi-curand 4096 2
echo DP===================

../../bin/linux/release/pi-curand 256 4
echo DP===================
../../bin/linux/release/pi-curand 1024 4
echo DP===================
../../bin/linux/release/pi-curand 4096 4
echo DP===================

../../bin/linux/release/pi-curand 256 8
echo DP===================
../../bin/linux/release/pi-curand 1024 8
echo DP===================
../../bin/linux/release/pi-curand 4096 8
echo DP===================
make clean
cd ..

cd ./pi-mystery
make clean
make sp
../../bin/linux/release/pi-mystery 256
echo SP===================
../../bin/linux/release/pi-mystery 1024
echo SP===================
../../bin/linux/release/pi-mystery 4096
echo SP===================
make clean

make dp
../../bin/linux/release/pi-mystery 256
echo DP===================
../../bin/linux/release/pi-mystery 1024
echo DP===================
../../bin/linux/release/pi-mystery 4096
echo DP===================
make clean
cd ..

cd ./pi-myrand
make sp
../../bin/linux/release/pi-myrand 256
echo SP===================
../../bin/linux/release/pi-myrand 1024
echo SP===================
../../bin/linux/release/pi-myrand 4096
echo SP===================
make clean

make dp
../../bin/linux/release/pi-myrand 256
echo DP===================
../../bin/linux/release/pi-myrand 1024
echo DP===================
../../bin/linux/release/pi-myrand 4096
echo DP===================
make clean
cd ..

cd ./pi-curand
make sp
../../bin/linux/release/pi-curand 256
echo SP===================
../../bin/linux/release/pi-curand 1024
echo SP===================
../../bin/linux/release/pi-curand 4096
echo SP===================
make clean

make dp
../../bin/linux/release/pi-curand 256
echo DP===================
../../bin/linux/release/pi-curand 1024
echo DP===================
../../bin/linux/release/pi-curand 4096
echo DP===================
make clean
cd ..

cd ./pi-curand-thrust
make sp
../../bin/linux/release/pi-curand-thrust 256
echo SP===================
../../bin/linux/release/pi-curand-thrust 1024
echo SP===================
../../bin/linux/release/pi-curand-thrust 4096
echo SP===================
make clean

make dp
../../bin/linux/release/pi-curand-thrust 256
echo DP===================
../../bin/linux/release/pi-curand-thrust 1024
echo DP===================
../../bin/linux/release/pi-curand-thrust 4096
echo DP===================
make clean
cd ..

echo FINISHED===================