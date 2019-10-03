from setuptools import setup

setup(
    name='tf-bspline',
    version='1.0.0',
    description='TensorFlow implementation of b-spline fitter',
    author='Peter Somers',
    license='MIT',
    long_description_content_type='text/markdown',
    #long_description=open('README.md').read(),
    packages=['tfbspline', 'tfbspline.util'],
    install_requires=['numpy', 'tensorflow']
)
