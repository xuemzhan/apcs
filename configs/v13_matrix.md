# v1.3 规模阶梯实验矩阵（第一关键路径：校准规模 + mapper 结构）
# A: affine      c200/e100  4B→1.7B
# B: affine_layer c200/e100 4B→1.7B   （参数少 8 倍）
# C: affine+task  c200/e100 4B→1.7B   （β=0.3, diag+α —— H3 判定）
# D: affine       c200/e100 4B→0.6B   （对偶泛化）
