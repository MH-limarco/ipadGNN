python main.py --methood $1 --model GCN --missing_rate 0.0

python main.py --methood $1 --model GCN --missing_rate 0.3 --fill fp
python main.py --methood $1 --model GCN --missing_rate 0.3 --fill apcfi
python main.py --methood $1 --model GCN --missing_rate 0.3 --fill zero

python main.py --methood $1 --model GCN --missing_rate 0.6 --fill fp
python main.py --methood $1 --model GCN --missing_rate 0.6 --fill apcfi
python main.py --methood $1 --model GCN --missing_rate 0.6 --fill zero

python main.py --methood $1 --model GCN --missing_rate 0.9 --fill fp
python main.py --methood $1 --model GCN --missing_rate 0.9 --fill apcfi
python main.py --methood $1 --model GCN --missing_rate 0.9 --fill zero

python main.py --methood $1 --model GCN --missing_rate 0.995 --fill fp
python main.py --methood $1 --model GCN --missing_rate 0.995 --fill apcfi
python main.py --methood $1 --model GCN --missing_rate 0.995 --fill zero

