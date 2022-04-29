build:
	g++ -O2 -msse4.1 utils.cpp fnn.cpp main.cpp -o main

build-arm:
	aarch64-linux-gnu-g++ -O2 -o main-arm -mcpu=cortex-a76 utils.cpp fnn.cpp main.cpp

build-huawei:
	g++ -O2 -o main-huawei -mcpu=cortex-a76 utils.cpp fnn.cpp main.cpp

run:
	qemu-arm -L /usr/arm-linux-gnueabihf/ main-arm

clean:
	rm -rf main main-arm
