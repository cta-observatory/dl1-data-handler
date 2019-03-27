conda config --add channels conda-forge
conda config --add channels anaconda
conda config --add channels cta-observatory
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
