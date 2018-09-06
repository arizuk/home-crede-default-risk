set -e

#python prev_single.py
python run.py
python run.py --kfold
csv=$(ls experiments/*csv|sort|tail -1)
kaggle competitions submit -c home-credit-default-risk -f ${csv} -m submit
