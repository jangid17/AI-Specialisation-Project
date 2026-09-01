# Contributing to AI Research Assistant

Thank you for your interest in contributing to the AI Research Assistant! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues
- Use the GitHub issue tracker
- Provide detailed descriptions of bugs
- Include steps to reproduce issues
- Attach relevant logs or screenshots

### Suggesting Features
- Open a discussion or issue
- Describe the feature and its benefits
- Consider implementation complexity
- Discuss with maintainers before major changes

### Code Contributions
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🛠️ Development Setup

### Prerequisites
- Python 3.12
- Git
- A free Groq API key (only for the features that call the language model; the tests need none)

### Setup Steps
```bash
# Clone your fork
git clone https://github.com/yourusername/ai-research-assistant.git
cd ai-research-assistant

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp env.example .env
# Edit .env and add your Groq key

# Run tests
python -m pytest -q
```

## 📝 Code Style Guidelines

### Python Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small

### Documentation
- Update README.md for new features
- Add inline comments for complex logic
- Update docstrings for API changes
- Include usage examples

### Testing
- Test new features thoroughly
- Add unit tests for core functionality
- Test error handling scenarios
- Verify compatibility with existing features

## 🚀 Pull Request Process

### Before Submitting
1. **Test your changes**: Ensure all functionality works
2. **Update documentation**: Update README and docstrings
3. **Check code style**: Follow PEP 8 guidelines
4. **Add tests**: Include tests for new features
5. **Update requirements**: Add new dependencies if needed

### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

## 🎯 Areas for Contribution

### High Priority
- **Performance Optimization**: Improve analysis speed
- **New Visualization Types**: Add more chart types
- **API Integration**: Connect to real academic databases
- **Export Formats**: Support more output formats

### Medium Priority
- **UI Improvements**: Better user interface
- **Error Handling**: More robust error recovery
- **Documentation**: Tutorials and guides
- **Testing**: Comprehensive test coverage

### Low Priority
- **Code Refactoring**: Clean up legacy code
- **Documentation**: Improve inline comments
- **Examples**: Add more usage examples
- **Templates**: Create project templates

## 🐛 Bug Reports

### Before Reporting
1. Check existing issues
2. Test with latest version
3. Verify it's not a configuration issue
4. Gather relevant information

### Bug Report Template
```markdown
**Bug Description**
Clear description of the bug

**Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- Version: [e.g., 1.0.0]

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

### Before Requesting
1. Check existing discussions
2. Consider implementation complexity
3. Think about user impact
4. Discuss with community

### Feature Request Template
```markdown
**Feature Description**
Clear description of the feature

**Use Case**
Why this feature would be useful

**Proposed Solution**
How you think it should work

**Alternatives**
Other approaches considered

**Additional Context**
Any other relevant information
```

## 📋 Code Review Process

### For Contributors
- Address review comments promptly
- Make requested changes
- Ask questions if unclear
- Be responsive to feedback

### For Reviewers
- Be constructive and helpful
- Focus on code quality
- Test the changes
- Provide clear feedback

## 🏷️ Release Process

### Version Numbering
- **Major**: Breaking changes
- **Minor**: New features
- **Patch**: Bug fixes

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version number updated
- [ ] Changelog updated
- [ ] Release notes prepared

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Email**: Direct contact for sensitive issues

### Response Times
- **Issues**: Within 48 hours
- **Pull Requests**: Within 72 hours
- **Discussions**: Within 24 hours

## 🎉 Recognition

### Contributors
- Listed in README.md
- Mentioned in release notes
- Invited to maintainer team (for significant contributions)

### Types of Contributions
- **Code**: Bug fixes, features, improvements
- **Documentation**: README, guides, tutorials
- **Testing**: Test cases, bug reports
- **Community**: Helping others, discussions

## 📄 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

## 🙏 Thank You

Thank you for contributing to the AI Research Assistant! Your contributions help make this project better for everyone.

---

**Questions?** Feel free to open a discussion or contact the maintainers directly.
