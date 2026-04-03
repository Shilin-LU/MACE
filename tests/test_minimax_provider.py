"""
Tests for MiniMax LLM provider support in MACE text_augmentation.
Run with: python -m pytest tests/test_minimax_provider.py -v
"""
import os
import sys
import types
import importlib
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal stubs so src/dataset.py can be imported without heavy ML deps
# ---------------------------------------------------------------------------

def _stub(name, attrs=None):
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod

_stub('torch')
_stub('torch.utils')
_stub('torch.utils.data', {'Dataset': object})
_stub('torchvision')
_stub('torchvision.transforms', {
    'ToTensor': object, 'Compose': object, 'Resize': object,
    'CenterCrop': object, 'Normalize': object, 'transforms': object,
})
_stub('torchvision.transforms.functional')
_stub('PIL', {'Image': MagicMock()})
_stub('omegaconf')

# stub regex to forward to built-in re
import re as _re
_regex = _stub('regex')
_regex.sub = _re.sub
_regex.compile = _re.compile

# stub openai before importing dataset
_mock_openai_cls = MagicMock()
_openai_mod = _stub('openai', {'OpenAI': _mock_openai_cls})

# stub src.cfr_utils (wildcard import)
_cfr = types.ModuleType('src.cfr_utils')
_cfr.__dict__['__all__'] = []
sys.modules['src.cfr_utils'] = _cfr

# Load src/dataset.py as a top-level module to avoid package resolution issues
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    'dataset_mod',
    os.path.join(os.path.dirname(__file__), '..', 'src', 'dataset.py'),
)
dataset_module = _ilu.module_from_spec(_spec)
# Inject stubs into the module's namespace before exec
dataset_module.__dict__['OpenAI'] = _mock_openai_cls
dataset_module.__dict__.update({k: v for k, v in _cfr.__dict__.items() if not k.startswith('__')})
_spec.loader.exec_module(dataset_module)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_completion(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


# ---------------------------------------------------------------------------
# Unit Tests: get_llm_client()
# ---------------------------------------------------------------------------

class TestGetLlmClientOpenAI(unittest.TestCase):
    def setUp(self):
        # Reset mock call counts
        _mock_openai_cls.reset_mock()

    def test_default_provider_is_openai(self):
        env = dict(os.environ)
        env.pop('LLM_PROVIDER', None)
        with patch.dict(os.environ, env, clear=True):
            _, model, provider = dataset_module.get_llm_client()
        self.assertEqual(provider, 'openai')
        self.assertEqual(model, 'gpt-3.5-turbo')

    def test_explicit_openai_provider(self):
        with patch.dict(os.environ, {'LLM_PROVIDER': 'openai'}):
            _, model, provider = dataset_module.get_llm_client()
        self.assertEqual(provider, 'openai')
        self.assertEqual(model, 'gpt-3.5-turbo')

    def test_unsupported_provider_raises(self):
        with patch.dict(os.environ, {'LLM_PROVIDER': 'unknown_xyz'}):
            with self.assertRaises(ValueError):
                dataset_module.get_llm_client()


class TestGetLlmClientMiniMax(unittest.TestCase):
    def setUp(self):
        _mock_openai_cls.reset_mock()

    def test_minimax_provider_selected(self):
        with patch.dict(os.environ, {'LLM_PROVIDER': 'minimax', 'MINIMAX_API_KEY': 'test-key'}):
            _, model, provider = dataset_module.get_llm_client()
        self.assertEqual(provider, 'minimax')

    def test_minimax_default_model_is_m27(self):
        env = {'LLM_PROVIDER': 'minimax', 'MINIMAX_API_KEY': 'k'}
        full_env = {**os.environ, **env}
        full_env.pop('MINIMAX_MODEL', None)
        with patch.dict(os.environ, full_env, clear=True):
            _, model, _ = dataset_module.get_llm_client()
        self.assertEqual(model, 'MiniMax-M2.7')

    def test_minimax_custom_model_via_env(self):
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'minimax',
            'MINIMAX_API_KEY': 'k',
            'MINIMAX_MODEL': 'MiniMax-M2.5-highspeed',
        }):
            _, model, _ = dataset_module.get_llm_client()
        self.assertEqual(model, 'MiniMax-M2.5-highspeed')

    def test_minimax_base_url_contains_minimax_io(self):
        with patch.dict(os.environ, {'LLM_PROVIDER': 'minimax', 'MINIMAX_API_KEY': 'k'}):
            dataset_module.get_llm_client()
        call_kwargs = _mock_openai_cls.call_args
        base_url_arg = str(call_kwargs)
        self.assertIn('minimax.io', base_url_arg)

    def test_all_minimax_model_variants_accepted(self):
        for m in dataset_module.MINIMAX_MODELS:
            with patch.dict(os.environ, {
                'LLM_PROVIDER': 'minimax',
                'MINIMAX_API_KEY': 'k',
                'MINIMAX_MODEL': m,
            }):
                _, model, _ = dataset_module.get_llm_client()
            self.assertEqual(model, m)


# ---------------------------------------------------------------------------
# Unit Tests: text_augmentation()
# ---------------------------------------------------------------------------

class TestTextAugmentationMiniMax(unittest.TestCase):

    def _fake_client(self, captions):
        completion = _make_completion(captions)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = completion
        return mock_client

    def test_minimax_temperature_passed_and_in_range(self):
        mock_client = self._fake_client('a ship at sea\na big ship')
        with patch.object(dataset_module, 'get_llm_client',
                          return_value=(mock_client, 'MiniMax-M2.7', 'minimax')):
            dataset_module.text_augmentation('ship', 'boat', 'object', num_text_augmentations=2)
        kw = mock_client.chat.completions.create.call_args[1]
        self.assertIn('temperature', kw, "MiniMax provider must pass temperature")
        self.assertGreater(kw['temperature'], 0.0)
        self.assertLessEqual(kw['temperature'], 1.0)

    def test_openai_no_extra_temperature(self):
        mock_client = self._fake_client('a ship at sea\na big ship')
        with patch.object(dataset_module, 'get_llm_client',
                          return_value=(mock_client, 'gpt-3.5-turbo', 'openai')):
            dataset_module.text_augmentation('ship', 'boat', 'object', num_text_augmentations=2)
        kw = mock_client.chat.completions.create.call_args[1]
        self.assertNotIn('temperature', kw, "OpenAI provider should not inject temperature")

    def test_minimax_correct_model_passed_to_api(self):
        mock_client = self._fake_client('a ship sailing\na big ship')
        with patch.object(dataset_module, 'get_llm_client',
                          return_value=(mock_client, 'MiniMax-M2.5-highspeed', 'minimax')):
            dataset_module.text_augmentation('ship', 'boat', 'object', num_text_augmentations=2)
        kw = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(kw['model'], 'MiniMax-M2.5-highspeed')

    def test_result_tuples_contain_erased_concept(self):
        mock_client = self._fake_client('a ship at sea\nthe ship is large\nbig ship sailing')
        with patch.object(dataset_module, 'get_llm_client',
                          return_value=(mock_client, 'MiniMax-M2.7', 'minimax')):
            cls_prompts, map_prompts = dataset_module.text_augmentation(
                'ship', 'boat', 'object', num_text_augmentations=2)
        self.assertGreater(len(cls_prompts), 0)
        for prompt, concept in cls_prompts:
            self.assertIn('ship', prompt)
            self.assertEqual(concept, 'ship')

    def test_mapping_prompts_use_mapping_concept(self):
        mock_client = self._fake_client('a ship at sea\nthe ship is large\nbig ship on water')
        with patch.object(dataset_module, 'get_llm_client',
                          return_value=(mock_client, 'MiniMax-M2.7', 'minimax')):
            cls_prompts, map_prompts = dataset_module.text_augmentation(
                'ship', 'boat', 'object', num_text_augmentations=2)
        self.assertEqual(len(cls_prompts), len(map_prompts))
        for prompt, concept in map_prompts:
            self.assertIn('boat', prompt)
            self.assertEqual(concept, 'boat')


# ---------------------------------------------------------------------------
# Unit Tests: clean_prompt()
# ---------------------------------------------------------------------------

class TestCleanPrompt(unittest.TestCase):
    def test_strips_leading_numbers(self):
        result = dataset_module.clean_prompt(['1. a ship', '2. another ship'])
        for r in result:
            self.assertFalse(r[:1].isdigit(), f"Expected no leading digit, got: {r!r}")

    def test_strips_double_quotes(self):
        result = dataset_module.clean_prompt(['"a ship"'])
        self.assertNotIn('"', result[0])

    def test_strips_whitespace(self):
        result = dataset_module.clean_prompt(['   a ship   '])
        self.assertEqual(result[0], 'a ship')

    def test_empty_list(self):
        result = dataset_module.clean_prompt([])
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Integration test (skipped unless MINIMAX_API_KEY is set)
# ---------------------------------------------------------------------------

class TestMiniMaxIntegration(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('MINIMAX_API_KEY'), 'MINIMAX_API_KEY not set')
    def test_live_client_creation(self):
        from openai import OpenAI as RealOpenAI
        client = RealOpenAI(
            base_url='https://api.minimax.io/v1',
            api_key=os.environ['MINIMAX_API_KEY'],
        )
        self.assertIsNotNone(client)

    @unittest.skipUnless(os.environ.get('MINIMAX_API_KEY'), 'MINIMAX_API_KEY not set')
    def test_live_text_augmentation_minimax(self):
        with patch.dict(os.environ, {'LLM_PROVIDER': 'minimax'}):
            cls_prompts, _ = dataset_module.text_augmentation(
                'ship', 'boat', 'object', num_text_augmentations=2)
        self.assertGreater(len(cls_prompts), 0)


if __name__ == '__main__':
    unittest.main()
