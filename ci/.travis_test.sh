#!/usr/bin/env bash

set -e

python --version
python -c "import pandas; print('pandas %s' % pandas.__version__)"
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"
python -c "import mlearner; print('mlearner %s' % mlearner.__version__)"

if [[ "$TRAVIS_OS_NAME" != "osx" ]]; then 


        if [[ "$COVERAGE" == "true" ]]; then

            if [[ "$IMAGE" == "true" ]]; then
                 PYTHONPATH='.' pytest -sv --cov=mlearner
            else
                 PYTHONPATH='.' pytest -sv --cov=mlearner --ignore=mlearner/image
            fi

        else
            if [[ "$IMAGE" == "true" ]]; then
                 PYTHONPATH='.' pytest -sv
            else
                 PYTHONPATH='.' pytest -sv --ignore=mlearner/image
            fi
        fi

else
     PYTHONPATH='.' pytest -sv --ignore=mlearner/plotting
fi
  

if [[ "$NOTEBOOKS" == "true" ]]; then
    cd docs

    if [[ "$IMAGE" == "true" ]]; then
      python make_api.py 
      find sources -name "*.ipynb" -exec jupyter nbconvert --to notebook --execute {} \;
    else      
      python make_api.py --ignore_packages "mlearner.image"
      find sources -name "*.ipynb" -not -path "sources/user_guide/image/*" -exec jupyter nbconvert --to notebook --execute {} \;
    
    fi
fi


