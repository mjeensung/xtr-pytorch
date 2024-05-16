import io
from setuptools import setup, find_packages

with open('readme.md', encoding='utf8') as f:
    readme = f.read()

with open('LICENSE', encoding='utf8') as f:
    license = f.read()

with open('requirements.txt', encoding='utf8') as f:
    reqs = f.read()

setup(
    name='xtr',
    version='1.0',
    description='Rethinking the Role of Token Retrieval in Multi-Vector Retrieval',
    long_description=readme,
    license=license,
    url='https://github.com/mjeensung/xtr_reimplementation',
    python_requires='>=3.8',
    packages=find_packages(include=['xtr', 'xtr.*']),
    install_requires=reqs.strip().split('\n'),
)