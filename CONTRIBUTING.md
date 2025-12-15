# Contributing to Meta-SpliceAI

Thank you for your interest in contributing to Meta-SpliceAI! This document provides guidelines and instructions for contributing to the project.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

---

## 🤝 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. Please treat all community members with respect.

### Expected Behavior

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Be patient and understanding
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discrimination, or intimidation
- Trolling, insulting comments, or personal attacks
- Publishing others' private information
- Other conduct that could reasonably be considered inappropriate

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- mamba or conda
- Familiarity with genomics and splice site prediction (helpful but not required)

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork**:
```bash
git clone https://github.com/YOUR_FORK/meta-spliceai.git
cd meta-spliceai
```

3. **Add upstream remote**:
```bash
git remote add upstream https://github.com/pleiadian53/meta-spliceai.git
```

4. **Create development environment**:
```bash
# Create environment
mamba create -n metaspliceai python=3.10 -y
mamba activate metaspliceai

# Install in editable mode
pip install -e .

# Install development dependencies
pip install pytest black flake8 mypy sphinx
```

5. **Verify installation**:
```bash
# Test CLI commands
meta-spliceai-run --help
meta-spliceai-eval --help

# Run tests
pytest tests/
```

---

## 🔄 Development Workflow

### 1. Create a Feature Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a new branch
git checkout -b feature/your-feature-name
```

**Branch naming conventions**:
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications

### 2. Make Your Changes

- Write clean, readable code
- Follow the project's coding standards
- Add tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

### 3. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "Add feature: description of your changes"
```

**Commit message guidelines**:
- Use present tense ("Add feature" not "Added feature")
- Be concise but descriptive
- Reference issues when applicable (#123)

### 4. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 5. Open a Pull Request

- Go to your fork on GitHub
- Click "New Pull Request"
- Select your branch
- Fill out the PR template
- Link related issues

---

## 🔍 Pull Request Process

### PR Checklist

Before submitting, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass (`pytest tests/`)
- [ ] New tests added for new functionality
- [ ] Documentation updated (if applicable)
- [ ] Commits are clean and well-organized
- [ ] PR description clearly explains the changes
- [ ] Related issues are linked

### Review Process

1. **Automated Checks**: CI/CD pipelines will run tests and linting
2. **Code Review**: Maintainers will review your code
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, your PR will be merged

### After Your PR is Merged

```bash
# Update your local main branch
git checkout main
git pull upstream main

# Delete your feature branch (optional)
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

---

## 📝 Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Imports**: Organized and grouped

### Code Formatting

Use `black` for automatic formatting:

```bash
# Format all Python files
black meta_spliceai/

# Format specific file
black meta_spliceai/module.py
```

### Linting

Use `flake8` for linting:

```bash
# Lint all code
flake8 meta_spliceai/

# Lint specific file
flake8 meta_spliceai/module.py
```

### Type Hints

Use type hints for function signatures:

```python
def predict_splice_sites(
    sequence: str,
    model_name: str = "openspliceai",
    threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """Predict splice sites in a sequence.
    
    Args:
        sequence: Input DNA sequence
        model_name: Base model to use
        threshold: Score threshold for predictions
        
    Returns:
        Dictionary with donor and acceptor predictions
    """
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Short description of function.
    
    Longer description if needed. Explain the purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When input is invalid
        
    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        True
    """
    pass
```

---

## 🧪 Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Mirror the package structure
- Use descriptive test names
- Test both success and failure cases

Example test structure:

```python
import pytest
from meta_spliceai.splice_engine import enhanced_evaluation

def test_calculate_metrics_basic():
    """Test basic metrics calculation."""
    evaluator = enhanced_evaluation.EnhancedEvaluator(
        predictions=mock_predictions,
        truth=mock_truth
    )
    metrics = evaluator.calculate_metrics()
    
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1

def test_calculate_metrics_empty_input():
    """Test metrics calculation with empty input."""
    evaluator = enhanced_evaluation.EnhancedEvaluator(
        predictions=[],
        truth=[]
    )
    
    with pytest.raises(ValueError, match="Empty input"):
        evaluator.calculate_metrics()
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_enhanced_evaluation.py

# Run specific test function
pytest tests/test_enhanced_evaluation.py::test_calculate_metrics_basic

# Run with coverage
pytest --cov=meta_spliceai --cov-report=html tests/

# Run with verbose output
pytest -v tests/
```

### Test Coverage

Aim for:
- **90%+ coverage** for core functionality
- **100% coverage** for critical components
- **Test edge cases** and error conditions

---

## 📚 Documentation

### Documentation Standards

- **Clear and concise**: Write for your audience
- **Code examples**: Include working examples
- **Keep updated**: Update docs with code changes
- **Cross-reference**: Link to related documentation

### Documentation Structure

```
docs/
├── installation/       # Installation guides
├── tutorials/          # Step-by-step tutorials
├── base_models/        # Base model documentation
├── training/           # Training workflows
├── testing/            # Testing procedures
└── development/        # Development guidelines
```

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build HTML documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

### Updating Documentation

1. **Inline documentation**: Update docstrings in code
2. **Package docs**: Update module-specific docs in `<package>/docs/`
3. **Project docs**: Update high-level docs in `docs/`
4. **README**: Update README.md if needed

---

## 🎯 Areas for Contribution

### High Priority

- [ ] Add support for new base models (SpliceGNN, Pangolin, etc.)
- [ ] Implement additional feature extraction methods
- [ ] Improve meta-model architectures
- [ ] Add comprehensive unit tests
- [ ] Write user tutorials and examples

### Medium Priority

- [ ] Optimize memory usage for large datasets
- [ ] Add support for alternative splicing modes
- [ ] Implement model ensemble methods
- [ ] Create visualization tools for predictions
- [ ] Add command-line progress indicators

### Good First Issues

- [ ] Fix typos in documentation
- [ ] Add type hints to existing functions
- [ ] Improve error messages
- [ ] Add docstring examples
- [ ] Write unit tests for utility functions

---

## 💡 Feature Requests

Have an idea for a new feature?

1. **Check existing issues**: See if it's already been suggested
2. **Open an issue**: Describe your feature request
3. **Discuss**: Engage with maintainers and community
4. **Implement**: Once approved, start working on it

---

## 🐛 Bug Reports

Found a bug?

1. **Check existing issues**: See if it's already reported
2. **Create minimal example**: Reproduce the bug
3. **Open an issue**: Include:
   - Description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Code example and error messages

---

## 🌟 Recognition

Contributors will be:
- Listed in the project's contributors page
- Mentioned in release notes
- Thanked publicly in the community

---

## 📞 Getting Help

- **Documentation**: Start with the [docs/](docs/)
- **Issues**: Search [existing issues](https://github.com/pleiadian53/meta-spliceai/issues)
- **Discussions**: Join [GitHub Discussions](https://github.com/pleiadian53/meta-spliceai/discussions)
- **Email**: Contact maintainers (if urgent)

---

## 📜 License

By contributing to Meta-SpliceAI, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Meta-SpliceAI! 🎉

Your contributions help advance genomics research and benefit the scientific community.

---

**Questions?** Feel free to ask in GitHub Discussions or open an issue.

