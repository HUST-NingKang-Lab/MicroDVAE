import unittest
from pathlib import Path


class RepoRenameTests(unittest.TestCase):
    def test_repo_branding_uses_microvqvae(self):
        readme = Path('README.md').read_text(encoding='utf-8')
        self.assertIn('# MicroVQVAE', readme)

        script_text = Path('scripts/tokenize_genome.py').read_text(encoding='utf-8')
        self.assertIn('MicroVQVAE', script_text)
        self.assertIn('from microvqvae.pipeline import tokenize_protein_fasta', script_text)

        self.assertTrue(Path('microvqvae').is_dir())


if __name__ == '__main__':
    unittest.main()
