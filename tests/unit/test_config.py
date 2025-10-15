"""
Unit tests for configuration system.

Fast tests that validate configuration loading and validation.
No GPU or heavy models required.
"""

import sys
from pathlib import Path
import tempfile

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import ModelConfig, DEFAULT_CONFIG, LIGHTWEIGHT_CONFIG


def test_config_creation():
    """Test basic config creation."""
    config = DEFAULT_CONFIG
    assert config is not None
    assert config.backbone.name == "resnet50"
    assert config.transformer.depth == 12
    print("✓ Config creation: PASS")


def test_config_validation():
    """Test config validation."""
    from config import TransformerConfig

    # Valid config
    config = TransformerConfig(depth=12, embed_dim=768, num_heads=12)
    assert config.embed_dim % config.num_heads == 0
    print("✓ Config validation: PASS")


def test_config_yaml_save_load():
    """Test YAML serialization."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        temp_path = f.name

    try:
        # Save
        config = DEFAULT_CONFIG
        config.to_yaml(temp_path)

        # Load
        loaded = ModelConfig.from_yaml(temp_path)
        assert loaded.backbone.name == config.backbone.name
        assert loaded.transformer.depth == config.transformer.depth
        print("✓ YAML save/load: PASS")
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_config_presets():
    """Test pre-configured profiles."""
    configs = [DEFAULT_CONFIG, LIGHTWEIGHT_CONFIG]

    for config in configs:
        assert config.transformer.depth > 0
        assert config.transformer.num_heads > 0
        assert config.training.batch_size > 0

    print("✓ Config presets: PASS")


def run_all_tests():
    """Run all config tests."""
    print("\n" + "=" * 60)
    print("Running Configuration Unit Tests")
    print("=" * 60)

    try:
        test_config_creation()
        test_config_validation()
        test_config_yaml_save_load()
        test_config_presets()

        print("\n" + "=" * 60)
        print("✅ All configuration tests passed!")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
