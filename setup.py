try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='guidedlda',
    version='0.0.1',
    url='https://github.com/ZhichaoZhong/guided-lda-wrapper',
    author='ZhichaoZhong',
    author_email='zzhong@wehkamp.nl',
    description='Guided LDA wrapper',
    install_requires = ['lda==1.1.0'],
    packages=["guidedlda"],
    python_requires='>=3.7',
)