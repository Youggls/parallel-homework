build:
	g++ utils.cpp fnn.cpp main.cpp -o main -g

build-arm:
	arm-linux-gnueabihf-g++ -O2 -o main-arm -mcpu=cortex-a76 -mfpu=neon-vfpv4 utils.cpp fnn.cpp main.cpp
build-huawei:
	g++ -O2 -o main-huawei -mcpu=cortex-a76 -mfpu=neon-vfpv4 utils.cpp fnn.cpp main.cpp

run:
	qemu-arm -L /usr/arm-linux-gnueabihf/ main-arm

clean:
	rm -rf main main-arm
