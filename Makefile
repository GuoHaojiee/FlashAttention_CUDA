# FlashAttention CUDA — Makefile
# 支持：CUDA 10.2+，sm_60（P100），x86_64 / ppc64le

# 自动探测 nvcc 路径（集群上通常在 /usr/local/cuda/bin 或 module 加载后在 PATH 里）
NVCC ?= $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)

# 目标架构：P100 = sm_60
ARCH     ?= sm_60

# CUDA 10.2 不需要 --std=c++17，默认 c++11 即可
CFLAGS    = -O2 -arch=$(ARCH) --shared -Xcompiler -fPIC -Xcompiler -O2

SRC_DIR   = src
OUT_DIR   = build

KERNELS   = naive_attention \
            flash_attention_v1 \
            flash_attention_v2 \
            flash_attention_v3

LIBS      = $(addprefix $(OUT_DIR)/, $(addsuffix .so, $(KERNELS)))

.PHONY: all clean check-nvcc

all: check-nvcc $(OUT_DIR) $(LIBS)

check-nvcc:
	@test -x "$(NVCC)" || (echo "错误：找不到 nvcc，请确认 CUDA 已加载（module load cuda）或设置 NVCC=/path/to/nvcc" && exit 1)
	@echo "使用编译器：$(NVCC)"
	@$(NVCC) --version | head -1

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(OUT_DIR)/%.so: $(SRC_DIR)/%.cu
	$(NVCC) $(CFLAGS) $< -o $@
	@echo "编译完成：$@"

clean:
	rm -rf $(OUT_DIR)

# 单独编译
k0: check-nvcc $(OUT_DIR)
	$(NVCC) $(CFLAGS) $(SRC_DIR)/naive_attention.cu -o $(OUT_DIR)/naive_attention.so

k1: check-nvcc $(OUT_DIR)
	$(NVCC) $(CFLAGS) $(SRC_DIR)/flash_attention_v1.cu -o $(OUT_DIR)/flash_attention_v1.so

k2: check-nvcc $(OUT_DIR)
	$(NVCC) $(CFLAGS) $(SRC_DIR)/flash_attention_v2.cu -o $(OUT_DIR)/flash_attention_v2.so

k3: check-nvcc $(OUT_DIR)
	$(NVCC) $(CFLAGS) $(SRC_DIR)/flash_attention_v3.cu -o $(OUT_DIR)/flash_attention_v3.so

# 如果集群用 module 管理环境，典型命令：
#   module load cuda/10.2
#   make ARCH=sm_60
#
# 如果 nvcc 不在 PATH：
#   make NVCC=/usr/local/cuda-10.2/bin/nvcc
#
# ppc64le 上编译无需额外参数，nvcc 会自动识别宿主架构
