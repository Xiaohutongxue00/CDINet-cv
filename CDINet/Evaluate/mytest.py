import numpy
import torch
gap = torch.nn.AdaptiveAvgPool2d(1)
# torch1 = torch.Tensor([[0, 0.5, 0.6, 0, 0], [0, 0.9, 0.8, 0, 0],
#                        [0, 0.9, 0.9, 0, 0], [0, 0.3, 0.4, 0, 0],
#                        [0, 0.2, 0.1, 0, 0]])
#
# torch2 = torch.Tensor([[0, 0, 0, 0, 0], [0.5, 0.9, 0.9, 0.3, 0.2],
#                        [0.6, 0.8, 0.9, 0.4, 0.1], [0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0]])
# print(torch1)
# print(torch2)
# torch1_gap = gap(torch1)
# torch2_gap = gap(torch2)
#
# print("___________RGB和Depth交集_________________")
# t = torch1.mul(torch2)
# print(t)
# t = torch.sqrt(t)
# print(t)
# t_gap = gap(t)
# print(f"torch1_gap = {torch1_gap}")
# print(f"torch2.gap = {torch2_gap}")
# print(f"t_gap = {t_gap}")
# rgb_w = t_gap / torch1_gap
# depth_w = t_gap / torch2_gap
#
# print(f"RGB weoght = {rgb_w}")
# print(f"depth weoght = {depth_w}")
#
#
#
# print("___________RGB和1-Depth交集_________________")
# t2 = torch1.mul(1-torch2)
# print(t2)
# t2 = torch.sqrt(t2)
# print(t2)
# t2_2 = torch1.mul(t2)
# print(t2_2)
#
# print("___________1-RGB和Depth交集_________________")
# t3 = torch2.mul(1-torch1)
# print(t3)
# t3 = torch.sqrt(t3)
# print(t3)


torch1 = torch.Tensor([[0, 1, 3, 5], [6, 0, 0, 8], [0, 7, 9, 1], [0, 0, 0, 0]])
torch2 = torch.Tensor([[0, 1, 3, 6], [0, 8, 0, 8], [0, 7, 0, 1], [0, 5, 4, 0]])
print(torch1)
print(torch2)
g1 = gap(torch1)
g2 = gap(torch2)
print(f"g1 = {g1}")
print(f"g2 = {g2}")

print('相减之后的结果')

x = torch1 - torch2
x = torch.abs(x)
print(x)
x_g = gap(x)
print(x_g)

z = x_g / (g1 + g2)
print(f"z = {z}")
print(f"权重 = {1 - z}")

