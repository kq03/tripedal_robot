import sympy as sp

# 定义符号变量，带下标
i = sp.symbols('i', integer=True)
a, L1, L2 = sp.symbols('a L1 L2')
theta_i1 = sp.Symbol('theta_{i1}')
theta_i2 = sp.Symbol('theta_{i2}')
alpha1, alpha2, alpha3 = sp.symbols('alpha1 alpha2 alpha3')
d1, d2, d3 = sp.symbols('d1 d2 d3')

def dh_matrix(a, alpha, d, theta):
    """
    计算单个DH参数的齐次变换矩阵（符号形式）
    """
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),                sp.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

# DH参数（全部用符号）
T01 = dh_matrix(a, -sp.pi/2, 0, 0)
T12 = dh_matrix(L1, 0, 0, theta_i1)
T23 = dh_matrix(L2, 0, 0, theta_i2)

# 总的齐次变换矩阵
T03 = T01 * T12 * T23

sp.pprint(T01, use_unicode=True)
sp.pprint(T12, use_unicode=True)
sp.pprint(T23, use_unicode=True)
sp.pprint(T03, use_unicode=True)