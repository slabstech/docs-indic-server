
sudo apt-get update

sudo apt-get install -y build-essential python3-dev libavcodec-dev libavformat-dev libswscale-dev cmake git
sudo apt-get install -y libavfilter-dev

git clone https://github.com/dmlc/dlpack.git
cd dlpack


sudo mkdir -p /usr/local/include/dlpack
sudo cp include/dlpack/dlpack.h /usr/local/include/dlpack/



git clone https://github.com/dmlc/dmlc-core.git

cd dmlc-core

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make

sudo mkdir -p /usr/local/include/dmlc
sudo cp -r ../include/dmlc/* /usr/local/include/dmlc/
sudo cp libdmlc.a /usr/local/lib/


git clone https://github.com/dmlc/decord.git
cd decord
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make

sudo mkdir -p /usr/local/include/dmlc
sudo cp -r ../include/dmlc/* /usr/local/include/dmlc/
sudo cp libdmlc.a /usr/local/lib/