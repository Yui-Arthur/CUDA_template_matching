NV := nvcc

# FLAGS := -DBLOCK_SIZE=$(BLOCK_SIZE)
template_matching.out:template_matching.cu PCC.h SSD.h
	$(NV) template_matching.cu -o template_matching.out

clean:
	rm template_matching.out