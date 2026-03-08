from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='deepspot',
      version='1.0.0',
      description='DeepSpot:',
      author='KalinNonchev',
      author_email='boo@foo.com',
      license='MIT License',
      long_description_content_type='text/markdown',
      long_description=open('README.md').read(),
      url="https://github.com/ratschlab/DeepSpot",
      packages=find_packages(),  # find packages
      include_package_data=True,
      # external packages as dependencies,
      install_requires=requirements,
      python_requires='>=3.9'
      )
