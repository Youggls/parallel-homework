build:
	g++ -O2 module.cpp test.cpp -o main

build-qemu:
	aarch64-linux-gnu-g++ -O2 -o main-qemu -mcpu=cortex-a76 module.cpp test.cpp

build-arm:
	g++ -O2 -o main-arm -mcpu=cortex-a76 module.cpp test.cpp

run-qemu:
	qemu-arm -L /usr/aarch64-linux-gnu/ main-qemu

clean:
	rm -rf main main-arm main-qemu
