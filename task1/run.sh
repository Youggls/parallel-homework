exec_name="main-huawei"

./${exec_name} -s false -d 512 -h 2048 -f 256 -o 32 -t 0
./${exec_name} -s true -d 512 -h 2048 -f 256 -o 32 -t 0

./${exec_name} -s false -d 512 -h 2048 -f 512 -o 32 -t 0
./${exec_name} -s true -d 512 -h 2048 -f 512 -o 32 -t 0

./${exec_name} -s false -d 512 -h 4096 -f 256 -o 32 -t 0
./${exec_name} -s true -d 512 -h 4096 -f 256 -o 32 -t 0

./${exec_name} -s false -d 512 -h 4096 -f 512 -o 32 -t 0
./${exec_name} -s true -d 512 -h 4096 -f 512 -o 32 -t 0

./${exec_name} -s false -d 1024 -h 4096 -f 256 -o 32 -t 0
./${exec_name} -s true -d 1024 -h 4096 -f 256 -o 32 -t 0

./${exec_name} -s false -d 1024 -h 4096 -f 512 -o 32 -t 0
./${exec_name} -s true -d 1024 -h 4096 -f 512 -o 32 -t 0

./${exec_name} -s false -d 2048 -h 4096 -f 256 -o 32 -t 0
./${exec_name} -s true -d 2048 -h 4096 -f 256 -o 32 -t 0

./${exec_name} -s false -d 2048 -h 4096 -f 512 -o 32 -t 0
./${exec_name} -s true -d 2048 -h 4096 -f 512 -o 32 -t 0

./${exec_name} -s false -d 2048 -h 8196 -f 256 -o 32 -t 0
./${exec_name} -s true -d 2048 -h 8196 -f 256 -o 32 -t 0

./${exec_name} -s false -d 2048 -h 8196 -f 512 -o 32 -t 0
./${exec_name} -s true -d 2048 -h 8196 -f 512 -o 32 -t 0

./${exec_name} -s false -d 2048 -h 8196 -f 1024 -o 32 -t 0
./${exec_name} -s true -d 2048 -h 8196 -f 1024 -o 32 -t 0
