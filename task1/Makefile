build:
	g++ -O2 -msse4.1 utils.cpp fnn.cpp main.cpp -o main

build-qemu:
	aarch64-linux-gnu-g++ -O2 -o main-qemu -mcpu=cortex-a76 utils.cpp fnn.cpp main.cpp

build-arm:
	g++ -O2 -o main-arm -mcpu=cortex-a76 utils.cpp fnn.cpp main.cpp

run-qemu:
	qemu-arm -L /usr/aarch64-linux-gnu/ main-qemu

clean:
	rm -rf main main-arm main-qemu
