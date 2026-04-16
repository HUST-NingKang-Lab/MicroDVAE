import unittest
from pathlib import Path


class ModelCommentStyleTests(unittest.TestCase):
    def test_model_comments_use_clean_english_only(self):
        text = Path('microvqvae/model.py').read_text(encoding='utf-8')
        banned_fragments = ['新增', '修复', '关键', '前期', '训练量化', '死码本', '便于', '一致性度量', '上下文解码路径', '原型旁路路径', '安全清理', '仅在有效位置计算']
        for fragment in banned_fragments:
            self.assertNotIn(fragment, text, f'Found banned comment fragment: {fragment}')


if __name__ == '__main__':
    unittest.main()
