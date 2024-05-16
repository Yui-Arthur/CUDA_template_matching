NV := nvcc

# FLAGS := -DTARGET=0

SSD ?= 1
PCC ?= 1
FLAGS := -DSSD_TEST=$(SSD) -DPCC_TEST=$(PCC)
# ifdef TARGET
# 	FLAGS := -DTARGET=$(TARGET)
# endif
build: main.cu PCC.h SSD.h
	$(NV) $(FLAGS) main.cu -o main.out

run: main.out
	bash ./test.sh $(TARGET)

PCC:
	make build PCC=1 SSD=0
	make run TARGET=$(TARGET)

SSD:
	make build PCC=0 SSD=1
	make run TARGET=$(TARGET)

main.out: build

clean:
	rm main.out