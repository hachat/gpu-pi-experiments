cd pi-curand
make clean
make sp
../../bin/linux/release/pi-curand 256 2
../../bin/linux/release/pi-curand 1024 2
../../bin/linux/release/pi-curand 4096 2

../../bin/linux/release/pi-curand 256 4
../../bin/linux/release/pi-curand 1024 4
../../bin/linux/release/pi-curand 4096 4

../../bin/linux/release/pi-curand 256 8
../../bin/linux/release/pi-curand 1024 8
../../bin/linux/release/pi-curand 4096 8

make clean

make dp
../../bin/linux/release/pi-curand 256 2
../../bin/linux/release/pi-curand 1024 2
../../bin/linux/release/pi-curand 4096 2

../../bin/linux/release/pi-curand 256 4
../../bin/linux/release/pi-curand 1024 4
../../bin/linux/release/pi-curand 4096 4

../../bin/linux/release/pi-curand 256 8
../../bin/linux/release/pi-curand 1024 8
../../bin/linux/release/pi-curand 4096 8
make clean
cd ..

cd ./pi-mystery
make clean
make sp
../../bin/linux/release/pi-mystery 256
../../bin/linux/release/pi-mystery 1024
../../bin/linux/release/pi-mystery 4096
make clean

make dp
../../bin/linux/release/pi-mystery 256
../../bin/linux/release/pi-mystery 1024
../../bin/linux/release/pi-mystery 4096
make clean
cd ..

cd ./pi-myrand
make clean
make sp
../../bin/linux/release/pi-myrand 256
../../bin/linux/release/pi-myrand 1024
../../bin/linux/release/pi-myrand 4096
make clean

make dp
../../bin/linux/release/pi-myrand 256
../../bin/linux/release/pi-myrand 1024
../../bin/linux/release/pi-myrand 4096
make clean
cd ..

cd ./pi-curand-thrust
make clean
make sp
../../bin/linux/release/pi-curand-thrust 256
../../bin/linux/release/pi-curand-thrust 1024
../../bin/linux/release/pi-curand-thrust 4096
make clean

make dp
../../bin/linux/release/pi-curand-thrust 256
../../bin/linux/release/pi-curand-thrust 1024
../../bin/linux/release/pi-curand-thrust 4096
make clean
cd ..