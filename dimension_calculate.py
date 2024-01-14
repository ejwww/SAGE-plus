import math
"以四层为例，T是层数"
T = 4

total_sum = 0
n_list = []
for t in range(0, T):
    a = math.sqrt(T - t + 1)
    total_sum += a
    n = int(128 * a / total_sum)
    n_list.append(n)

# 计算缩放比例
scale = 128 / sum(n_list)

# 缩放每个n值
n_list = [int(n * scale) for n in n_list]

# 调整最后一个n值以确保总和为32
n_list[-1] += 128 - sum(n_list)

# 判断最后一个n值是否为1，如果是，则后续的n值都设为1
if n_list[-1] == 1:
    n_list[-1:] = [1] * (T - len(n_list) + 1)

# 打印每个n值和总和
print("n values:", n_list)
print("Sum of n values:", sum(n_list))

