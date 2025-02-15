pip uninstall -y deepspot
rm -rf dist/ aestetik.egg-info/ build/
python setup.py install --user
