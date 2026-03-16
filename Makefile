# FlashAttention CUDA — Makefile
# 支持：CUDA 10.2+，sm_60（P100），x86_64 / ppc64le

NVCC  ?= $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
ARCH  ?= sm_60

SOFLAGS = -O2 -arch=$(ARCH) --shared -Xcompiler -fPIC
BINFLAGS = -O2 -arch=$(ARCH)

SRC_DIR  = src
OUT_DIR  = build

KERNELS  = naive_attention \
           flash_attention_v1 \
           flash_attention_v2 \
           flash_attention_v3 \
           flash_attention_v4

LIBS     = $(addprefix $(OUT_DIR)/, $(addsuffix .so, $(KERNELS)))
BENCH    = $(OUT_DIR)/benchmark

.PHONY: all bench clean check-nvcc

# 默认目标：编译所有 .so + benchmark 可执行文件
all: check-nvcc $(OUT_DIR) $(LIBS) $(BENCH)

# 只编译 benchmark 可执行文件（依赖所有 kernel .cu 源文件）
bench: check-nvcc $(OUT_DIR)
	$(NVCC) $(BINFLAGS) \
	    bench/benchmark.cu \
	    $(SRC_DIR)/naive_attention.cu \
	    $(SRC_DIR)/flash_attention_v1.cu \
	    $(SRC_DIR)/flash_attention_v2.cu \
	    $(SRC_DIR)/flash_attention_v3.cu \
	    $(SRC_DIR)/flash_attention_v4.cu \
	    -o $(BENCH)
	@echo "编译完成：$(BENCH)"
	@echo "运行：./$(BENCH)"

# benchmark 可执行文件也作为 all 的依赖，规则同上
$(BENCH): bench/benchmark.cu $(addprefix $(SRC_DIR)/, $(addsuffix .cu, $(KERNELS)))
	$(NVCC) $(BINFLAGS) \
	    bench/benchmark.cu \
	    $(SRC_DIR)/naive_attention.cu \
	    $(SRC_DIR)/flash_attention_v1.cu \
	    $(SRC_DIR)/flash_attention_v2.cu \
	    $(SRC_DIR)/flash_attention_v3.cu \
	    $(SRC_DIR)/flash_attention_v4.cu \
	    -o $(BENCH)

# 各 kernel 的独立 .so（保留，供其他工具/ctypes 使用）
$(OUT_DIR)/%.so: $(SRC_DIR)/%.cu
	$(NVCC) $(SOFLAGS) $< -o $@

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

check-nvcc:
	@test -x "$(NVCC)" || \
	    (echo "错误：找不到 nvcc，请 'module load cuda' 或设置 NVCC=/path/to/nvcc" && exit 1)
	@$(NVCC) --version | head -1

clean:
	rm -rf $(OUT_DIR)

# 单独编译某个 kernel .so
k0: check-nvcc $(OUT_DIR); $(NVCC) $(SOFLAGS) $(SRC_DIR)/naive_attention.cu    -o $(OUT_DIR)/naive_attention.so
k1: check-nvcc $(OUT_DIR); $(NVCC) $(SOFLAGS) $(SRC_DIR)/flash_attention_v1.cu -o $(OUT_DIR)/flash_attention_v1.so
k2: check-nvcc $(OUT_DIR); $(NVCC) $(SOFLAGS) $(SRC_DIR)/flash_attention_v2.cu -o $(OUT_DIR)/flash_attention_v2.so
k3: check-nvcc $(OUT_DIR); $(NVCC) $(SOFLAGS) $(SRC_DIR)/flash_attention_v3.cu -o $(OUT_DIR)/flash_attention_v3.so
k4: check-nvcc $(OUT_DIR); $(NVCC) $(SOFLAGS) $(SRC_DIR)/flash_attention_v4.cu -o $(OUT_DIR)/flash_attention_v4.so

# 集群常用命令：
#   module load cuda/10.2
#   make bench                          # 只编译 + 运行 benchmark
#   make NVCC=/usr/local/cuda-10.2/bin/nvcc ARCH=sm_60
