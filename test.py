import unittest
from your_module import distance, generate_nodes, select_heads  # 假设你的代码在your_module.py文件中

class TestWSNMethods(unittest.TestCase):

    def test_distance(self):
        # 测试distance函数
        self.assertAlmostEqual(distance([0, 0], [1, 1]), np.sqrt(2))
        self.assertAlmostEqual(distance([0, 0], [0, 0]), 0)
        self.assertAlmostEqual(distance([1, 2], [3, 4]), np.sqrt(8))

    def test_generate_nodes(self):
        # 测试generate_nodes函数
        np.random.seed(0)  # 设置随机数种子以保证结果可预测
        nodes, sign_point = generate_nodes(10)
        self.assertEqual(len(nodes), 10)
        self.assertEqual(len(sign_point), 10)
        self.assertEqual(sign_point, [0] * 10)

    def test_select_heads(self):
        # 测试select_heads函数
        np.random.seed(0)  # 设置随机数种子以保证结果可预测
        nodes = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        flags = [0, 0, 0, 0]
        r = 1
        P = 0.5
        heads, members = select_heads(r, nodes, flags, P)
        self.assertEqual(len(heads), 2)  # 根据随机数种子和P值，预期有2个簇首
        self.assertEqual(len(members), 2)  # 剩余的是成员节点

# 运行单元测试
if __name__ == '__main__':
    unittest.main()
